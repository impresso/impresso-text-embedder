"""Local-inference (CaaS) variant of :mod:`research.query_generate`.

Mirrors :mod:`research.query_generate` byte-for-byte on the
non-LLM-client surface — same prompts, same Pydantic schema, same
verbatim-anchor verification, same study-YAML-driven CLI, same output
schema — but swaps the EPFL RCP AIaaS endpoint (LangChain ``ChatOpenAI``)
for a local Hugging-Face ``transformers`` ``AutoModelForCausalLM``
loaded inside the production Run:AI container. The two scripts emit
the same ``queries.jsonl.bz2`` shape; downstream eval cannot tell
which path produced the queries beyond the ``gen_endpoint`` field
(``"local:transformers"`` vs ``"https://inference.rcp.epfl.ch/v1"``).

Optimisation recipe matches the production embedder's bf16 strategy
(see ``.history/gpu-throughput/notes.md``): bf16 weights via
``torch_dtype=torch.bfloat16`` (``torch_dtype`` over the newer
``dtype=`` kwarg keeps us compatible with transformers 4.x, the
container's pin) plus ``attn_implementation`` set to either
``"sdpa"`` (default — built into PyTorch, dispatches to Flash
Attention v2 on Ampere/Hopper without an extra wheel) or
``"flash_attention_2"`` (opt-in, requires a separate ``flash-attn``
install on top of the current Dockerfile). ``"eager"`` is available
as the no-fast-path baseline.

Why CaaS as well as AIaaS: the AIaaS endpoint has a per-key parallel
cap and is shared infrastructure; a Run:AI container with a single
H100 can run the full study self-contained, mirrors the production
embed sweep's submission shape, and isolates the experiment from
AIaaS-side throttling drift.

Common helpers (prompts, ``QueryOutput``, ``CorpusRecord``,
``verify_references``, ``_plan_jobs``, ``read_corpus_shard``,
``write_queries``) are imported directly from
:mod:`research.query_generate` so the prompt and parsing contracts
stay single-source-of-truth.

Design rationale, rejected alternatives, and the "flash-attn install
follow-up" note in ``.progress/query-generation/notes.md`` (CaaS
section).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

from impresso_text_embedder.research._io import staged_input, staged_output
from impresso_text_embedder.research.query_generate import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_QUERIES_PER_BUCKET,
    DEFAULT_TEMPERATURE,
    GenerationConfig,
    GenerationStats,
    Job,
    JobResult,
    Query,
    QueryOutput,
    _format_stats,
    _plan_jobs,
    build_system_prompt,
    build_user_message,
    read_corpus_shard,
    verify_references,
    write_queries,
)
from impresso_text_embedder.research.study_config import (
    CORPUS_FILENAME,
    QUERIES_FILENAME,
    load_study_config,
)

log = logging.getLogger(__name__)


# Local-inference defaults. These are the knobs the AIaaS path doesn't
# need; we keep them out of GenerationConfig so the existing
# AIaaS-side dataclass stays unchanged.
DEFAULT_DTYPE: str = "bf16"
# sdpa is built into PyTorch and dispatches to Flash Attention v2 on
# Ampere/Hopper without requiring the separate flash-attn wheel — works
# with the current Dockerfile out of the box. flash_attention_2 is the
# explicit opt-in once flash-attn is added to the image.
DEFAULT_ATTENTION: str = "sdpa"
DEFAULT_MAX_RETRIES: int = 3
# Endpoint string recorded on every emitted Query to distinguish CaaS
# runs from AIaaS runs in the downstream eval. Not a real URL.
LOCAL_ENDPOINT: str = "local:transformers"

_DTYPE_MAP: dict[str, Any] = {}  # populated lazily — torch import is heavy


def _resolve_dtype(name: str):  # noqa: ANN201 — torch dtype, lazy import
    import torch

    if not _DTYPE_MAP:
        _DTYPE_MAP.update(
            bf16=torch.bfloat16,
            fp16=torch.float16,
            fp32=torch.float32,
        )
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype {name!r}; expected one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


# ---------------------------------------------------------------------------
# Output parsing — JSON with optional code-fence wrapper
# ---------------------------------------------------------------------------


# Strip a ```json … ``` (or plain ```) fence the model may wrap around
# the JSON object. The system prompt asks for raw JSON, so most
# completions arrive without fences, but we accept either form.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_query_output(text: str) -> QueryOutput | None:
    """Best-effort parse of a raw model completion into :class:`QueryOutput`.

    Mirrors what LangChain's ``with_structured_output(method="json_mode")``
    does on the AIaaS side, minus the strict ``json_mode`` API contract
    (which transformers ``.generate()`` doesn't have): strip an optional
    ```json fence, locate the first ``{ … }`` object in the completion
    (handles the rare leading-prose case), parse via ``orjson``, then
    validate against the shared :class:`QueryOutput` Pydantic schema.

    Returns ``None`` on any failure so the retry loop in
    :func:`generate_one_local` can try again with a fresh sample.
    """
    s = text.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    s = s[start : end + 1]
    try:
        payload = orjson.loads(s)
    except orjson.JSONDecodeError:
        return None
    try:
        return QueryOutput.model_validate(payload)
    except Exception:  # noqa: BLE001 — pydantic.ValidationError + edge cases
        return None


# ---------------------------------------------------------------------------
# Local LLM client — transformers AutoModelForCausalLM + chat template
# ---------------------------------------------------------------------------


def load_local_llm(
    model_name: str,
    *,
    dtype: str = DEFAULT_DTYPE,
    attn_implementation: str = DEFAULT_ATTENTION,
    device_map: str | dict | None = "auto",
):
    """Load a Qwen-style causal LM with bf16 + fast-attention defaults.

    The recipe mirrors the production embedder
    (``model.load_model``): bf16 weights, fast attention path on by
    default, model in eval mode. ``device_map="auto"`` lets HF pick the
    visible CUDA device — for the single-GPU-per-container Run:AI
    pattern this resolves to ``cuda:0``.

    ``attn_implementation="flash_attention_2"`` raises if the
    ``flash-attn`` wheel isn't installed; the default ``"sdpa"`` works
    with the current Dockerfile because PyTorch's
    ``scaled_dot_product_attention`` dispatches to FA2 internally on
    bf16 + Ampere/Hopper.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(
        "loading local LLM %s (dtype=%s attn=%s device_map=%s)",
        model_name,
        dtype,
        attn_implementation,
        device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Causal-LM batched generation needs left padding so the prompt's
    # final token sits at the rightmost position before we start
    # decoding. Set the pad token to eos when the model didn't ship one
    # (Qwen3 does, but be defensive).
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_resolve_dtype(dtype),
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()
    log.info(
        "loaded %s on device=%s dtype=%s attn=%s",
        model_name,
        next(model.parameters()).device,
        next(model.parameters()).dtype,
        attn_implementation,
    )
    # torch is imported only for the side-effect that the model is on a
    # CUDA device when one is visible; nothing else needs it here.
    del torch
    return model, tokenizer


def generate_completion(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.8,
    top_k: int = 20,
) -> str:
    """Run one chat-template + ``model.generate`` round and return the
    model's reply (system + user messages stripped from the output).

    Sampling defaults (``top_p=0.8``, ``top_k=20``) match the
    Qwen3-30B-A3B-Instruct-2507 recommended settings from the model
    card. ``temperature == 0`` switches to greedy decoding (the
    ``do_sample=False`` path), so the script can be run in
    deterministic mode for reproducibility checks.
    """
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Strip the prompt prefix; only return what the model produced.
    new_ids = out_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-job generation — synchronous (one GPU, no concurrency)
# ---------------------------------------------------------------------------


def generate_one_local(
    job: Job,
    cfg: GenerationConfig,
    *,
    generate_fn: Callable[[list[dict[str, str]]], str],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> JobResult:
    """Generate one query for ``job`` via a local model.

    ``generate_fn`` is injected so this function can be unit-tested
    without loading real weights — production wires it to a
    closure over :func:`generate_completion` bound to the loaded
    model + tokenizer + sampling knobs.

    Retries on parse failure (model returned non-JSON or didn't match
    the :class:`QueryOutput` schema). Real generation errors propagate
    out of ``generate_fn`` and are tagged ``error_kind="api"`` to
    mirror the AIaaS path's failure taxonomy — the eval doesn't
    distinguish which side broke.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(job.query_type, job.record.lg)},
        {
            "role": "user",
            "content": build_user_message(job.record, job.bucket_label, job.bucket_text),
        },
    ]

    parsed: QueryOutput | None = None
    for attempt in range(max(1, max_retries)):
        try:
            raw = generate_fn(messages)
        except Exception as exc:  # noqa: BLE001 — surfacing per-call failure
            log.warning(
                "local generate error for ci_id=%s bucket=%s type=%s: %s",
                job.record.ci_id,
                job.bucket_label,
                job.query_type,
                exc,
            )
            return JobResult(query=None, error_kind="api")
        parsed = parse_query_output(raw)
        if parsed is not None:
            break
        log.debug(
            "parse failed (attempt %d/%d) for ci_id=%s bucket=%s type=%s; raw[:200]=%r",
            attempt + 1,
            max_retries,
            job.record.ci_id,
            job.bucket_label,
            job.query_type,
            raw[:200],
        )
    if parsed is None:
        return JobResult(query=None, error_kind="api")

    verified, not_found, oob = verify_references(
        job.record.ft, parsed.references, job.bucket_range
    )
    if not verified:
        return JobResult(
            query=None,
            error_kind="no_refs",
            refs_not_found=not_found,
            refs_out_of_bucket=oob,
        )

    query = Query(
        query_id=(
            f"{job.record.ci_id}__{job.bucket_label}__{job.query_type}"
            f"__{job.sample_idx:02d}"
        ),
        ci_id=job.record.ci_id,
        lg=job.record.lg,
        query_text=parsed.query.strip(),
        query_type=job.query_type,
        references=tuple(verified),
        position_bucket=job.bucket_label,
        position_chars=job.bucket_range,
        gen_model=cfg.model,
        gen_endpoint=LOCAL_ENDPOINT,
        ts=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        study_name=cfg.study_name,
        study_config_sha=cfg.study_config_sha,
    )
    return JobResult(query=query, refs_not_found=not_found, refs_out_of_bucket=oob)


# ---------------------------------------------------------------------------
# Sync run loop — single GPU, single thread
# ---------------------------------------------------------------------------


def generate_queries_local(
    records: Sequence,
    cfg: GenerationConfig,
    *,
    generate_fn: Callable[[list[dict[str, str]]], str],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[list[Query], GenerationStats]:
    """Drive generation across all (doc, bucket, query_type) jobs sequentially.

    No semaphore / asyncio: one GPU, one stream of generations. The
    AIaaS path runs concurrent because the bottleneck is per-key TCP
    parallelism, not GPU compute; here the GPU is the bottleneck so a
    single inflight request is the right shape.
    """
    stats = GenerationStats(corpus_records=len(records))
    jobs = _plan_jobs(records, cfg)
    if not jobs:
        return [], stats

    log.info(
        "planned: corpus_records=%d jobs=%d buckets=%s queries_per_bucket=%d "
        "model=%s",
        stats.corpus_records,
        len(jobs),
        list(cfg.position_buckets),
        cfg.queries_per_bucket,
        cfg.model,
    )

    queries: list[Query] = []
    pbar = tqdm(total=len(jobs), file=sys.stderr, disable=None, desc="generate", unit="q")
    try:
        for job in jobs:
            result = generate_one_local(
                job, cfg, generate_fn=generate_fn, max_retries=max_retries
            )
            stats.attempts += 1
            stats.refs_not_found += result.refs_not_found
            stats.refs_out_of_bucket += result.refs_out_of_bucket
            if result.query is not None:
                queries.append(result.query)
                stats.queries_kept += 1
                q = result.query
                stats.by_bucket[q.position_bucket] = (
                    stats.by_bucket.get(q.position_bucket, 0) + 1
                )
                stats.by_lg[q.lg] = stats.by_lg.get(q.lg, 0) + 1
                stats.by_query_type[q.query_type] = (
                    stats.by_query_type.get(q.query_type, 0) + 1
                )
            elif result.error_kind == "api":
                stats.api_errors += 1
            elif result.error_kind == "no_refs":
                stats.no_refs_returned += 1
            pbar.update(1)
    finally:
        pbar.close()
    return queries, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-research-query-generate-local",
        description=(
            "Generate position-stratified synthetic queries from the "
            "chunking-eval corpus shard via a locally loaded LLM "
            "(transformers + bf16 + sdpa/flash-attn-2). Same prompts, "
            "same output schema, same study YAML as the AIaaS variant; "
            "the only difference is where the model lives."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Study YAML (e.g. configs/research/study-v1.yaml). Source "
            "of truth for corpus/output paths and query_generation.* "
            "defaults; per-flag CLI args still override field-by-field."
        ),
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "skip the S3 upload; the queries shard lands at the study's "
            "local mirror (paths.local_root + queries.jsonl.bz2)."
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["bf16", "fp16", "fp32"],
        help="weights/activations dtype passed to torch_dtype",
    )
    p.add_argument(
        "--attention",
        default=DEFAULT_ATTENTION,
        choices=["sdpa", "flash_attention_2", "eager"],
        help=(
            "attn_implementation passed to from_pretrained. sdpa works "
            "out of the box with the current Dockerfile and dispatches "
            "to FA2 on bf16+Ampere/Hopper; flash_attention_2 requires a "
            "separate flash-attn install on top of the image."
        ),
    )
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    p.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="max attempts per job before declaring an api/parse failure",
    )
    p.add_argument(
        "--queries-per-bucket",
        type=int,
        default=DEFAULT_QUERIES_PER_BUCKET,
        help=(
            "multiplicity per (doc, bucket, query_type) cell; default 1. "
            "Raise to N for N x compute and N x statistical power."
        ),
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    return p


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a runtime dep
        return
    load_dotenv()


@dataclasses.dataclass(frozen=True)
class LocalKnobs:
    """Local-inference-only knobs that don't fit GenerationConfig.

    Kept tiny on purpose: the shared :class:`GenerationConfig` carries
    every field the prompts + planner read, and these three fields are
    the only ones unique to the CaaS path.
    """

    dtype: str = DEFAULT_DTYPE
    attention: str = DEFAULT_ATTENTION
    max_retries: int = DEFAULT_MAX_RETRIES


def config_from_args(
    args: argparse.Namespace, study_cfg
) -> tuple[GenerationConfig, LocalKnobs]:
    """Translate CLI args + study YAML into a (shared, local-only) pair.

    The shared :class:`GenerationConfig` reuses the AIaaS fields with
    their AIaaS-specific entries (``api_key``, ``endpoint``,
    ``max_parallel``, ``request_timeout_s``, ``retry_attempts``) left
    at neutral defaults — the local path simply doesn't read them. Per
    the existing pattern: when the CLI flag matches its argparse
    default, the YAML wins; otherwise the CLI overrides.
    """
    study_qg = study_cfg.query_generation

    def _pick(cli_val: object, default_val: object, cfg_val: object) -> object:
        if cli_val == default_val:
            return cfg_val
        return cli_val

    model = _pick(args.model, DEFAULT_MODEL, study_qg.model)
    temperature = float(
        _pick(args.temperature, DEFAULT_TEMPERATURE, study_qg.temperature)
    )
    max_output_tokens = int(
        _pick(
            args.max_output_tokens,
            DEFAULT_MAX_OUTPUT_TOKENS,
            study_qg.max_output_tokens,
        )
    )
    queries_per_bucket = int(
        _pick(
            args.queries_per_bucket,
            DEFAULT_QUERIES_PER_BUCKET,
            study_qg.queries_per_bucket,
        )
    )

    gen = GenerationConfig(
        model=str(model),
        endpoint=LOCAL_ENDPOINT,
        api_key="",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        limit=args.limit,
        study_name=study_cfg.study.name,
        study_config_sha=study_cfg.config_sha,
        position_buckets=tuple(study_qg.position_buckets),
        queries_per_bucket=queries_per_bucket,
    )
    knobs = LocalKnobs(
        dtype=args.dtype,
        attention=args.attention,
        max_retries=int(args.max_retries),
    )
    return gen, knobs


def _run(
    cfg: GenerationConfig,
    knobs: LocalKnobs,
    *,
    bucket: str,
    corpus_key: str,
    out_key: str,
    out_local_mirror: Path,
    upload: bool,
) -> int:
    with staged_input(bucket, corpus_key) as corpus_path:
        records = read_corpus_shard(corpus_path)
    if cfg.limit is not None:
        records = records[: cfg.limit]
    log.info("loaded %d corpus records", len(records))

    model, tokenizer = load_local_llm(
        cfg.model, dtype=knobs.dtype, attn_implementation=knobs.attention
    )

    def generate_fn(messages: list[dict[str, str]]) -> str:
        return generate_completion(
            model,
            tokenizer,
            messages,
            max_new_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    queries, stats = generate_queries_local(
        records, cfg, generate_fn=generate_fn, max_retries=knobs.max_retries
    )
    log.info("generation stats: %s", _format_stats(stats))

    with staged_output(bucket, out_key, out_local_mirror, upload=upload) as out_path:
        write_queries(queries, out_path)
        log.info("queries written: %s (%d queries)", out_path, len(queries))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = args.log_level.upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _load_env()

    study_cfg = load_study_config(args.config)
    cfg, knobs = config_from_args(args, study_cfg)

    bucket = study_cfg.s3.bucket
    corpus_key = study_cfg.s3_key(CORPUS_FILENAME)
    out_key = study_cfg.s3_key(QUERIES_FILENAME)
    out_local_mirror = study_cfg.local_path(QUERIES_FILENAME)

    log.info(
        "query-generate-local start: study=%s model=%s dtype=%s attn=%s "
        "corpus=s3://%s/%s output=s3://%s/%s upload=%s",
        cfg.study_name or "(none)",
        cfg.model,
        knobs.dtype,
        knobs.attention,
        bucket,
        corpus_key,
        bucket,
        out_key,
        "no" if args.no_upload else "yes",
    )
    return _run(
        cfg,
        knobs,
        bucket=bucket,
        corpus_key=corpus_key,
        out_key=out_key,
        out_local_mirror=out_local_mirror,
        upload=not args.no_upload,
    )


__all__ = [
    "DEFAULT_ATTENTION",
    "DEFAULT_DTYPE",
    "DEFAULT_MAX_RETRIES",
    "LOCAL_ENDPOINT",
    "LocalKnobs",
    "build_parser",
    "config_from_args",
    "generate_completion",
    "generate_one_local",
    "generate_queries_local",
    "load_local_llm",
    "main",
    "parse_query_output",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
