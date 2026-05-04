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
import time
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
# Default ``--batch-size`` is 1 (single-stream) so a laptop / smaller-GPU
# run is the safe default. The Run:AI submit target overrides to 8 for
# H100 80GB (Qwen3-30B-A3B in bf16 ≈ 60 GB weights + ~1 GB / sequence
# KV cache at 10k context → batch=8 fits with ~12 GB of headroom). See
# ``.progress/query-generation/notes.md`` (CaaS section) for the
# memory math.
DEFAULT_BATCH_SIZE: int = 1
# Cadence (in batches) of the periodic progress log line. 50 is the
# right granularity for the Run:AI submit defaults: at study-A-fit
# ~900 batches @ batch=4 → ~18 progress lines over a ~1-2 h run. Tune
# via ``log_every_n`` on :func:`generate_queries_local`.
DEFAULT_PROGRESS_LOG_EVERY_N_BATCHES: int = 50
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
    device: str | None = None,
):
    """Load a Qwen-style causal LM with bf16 + fast-attention defaults.

    The recipe mirrors the production embedder
    (``model.load_model``): bf16 weights, fast attention path on by
    default, model in eval mode. The model lands on ``cuda`` when a
    CUDA device is visible — explicit ``.to(device)`` rather than
    ``device_map="auto"`` because ``device_map`` requires the
    ``accelerate`` package, which is not currently bundled in the
    production NGC image and would have to be added as a runtime
    dep. The single-GPU-per-container Run:AI pattern doesn't need
    accelerate's auto-sharding; one CUDA device, one ``.to`` call.

    Two-step CPU→GPU placement does mean a transient peak of ~2×
    weights in host RAM on load (60 GB for Qwen3-30B-A3B in bf16);
    the production Run:AI pod default of 64 GB+ swallows that fine.
    Promote to ``device_map="auto"`` once ``accelerate`` lands in
    the image (CaaS open item OC4 in the query-generation notes).

    ``attn_implementation="flash_attention_2"`` raises if the
    ``flash-attn`` wheel isn't installed; the default ``"sdpa"`` works
    with the current Dockerfile because PyTorch's
    ``scaled_dot_product_attention`` dispatches to FA2 internally on
    bf16 + Ampere/Hopper.

    ``dtype=`` (not ``torch_dtype=``) per the transformers 4.55+
    rename. Both kwargs work on transformers ≥4.45 (the deprecation
    started in 4.55), but ``dtype`` silences the runtime warning.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(
        "loading local LLM %s (dtype=%s attn=%s device=%s)",
        model_name,
        dtype,
        attn_implementation,
        device,
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
        dtype=_resolve_dtype(dtype),
        attn_implementation=attn_implementation,
    )
    model.eval()
    model.to(device)
    log.info(
        "loaded %s on device=%s dtype=%s attn=%s",
        model_name,
        next(model.parameters()).device,
        next(model.parameters()).dtype,
        attn_implementation,
    )
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

    Single-prompt convenience wrapper around
    :func:`generate_completions_batched`. Used by the parse-retry
    fallback inside :func:`generate_queries_local` (one bad
    completion in a batch is retried as a single-job call rather
    than re-running the whole batch) and by the unit-test path.

    Sampling defaults (``top_p=0.8``, ``top_k=20``) match the
    Qwen3-30B-A3B-Instruct-2507 recommended settings from the model
    card. ``temperature == 0`` switches to greedy decoding (the
    ``do_sample=False`` path), so the script can be run in
    deterministic mode for reproducibility checks.
    """
    return generate_completions_batched(
        model,
        tokenizer,
        [messages],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )[0]


def generate_completions_batched(
    model: Any,
    tokenizer: Any,
    messages_batch: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.8,
    top_k: int = 20,
) -> list[str]:
    """Run one ``model.generate`` round over ``B`` chat prompts in
    parallel; return ``B`` decoded reply strings (prompt prefixes
    stripped).

    Left-padded so the rightmost token of every prompt aligns at the
    same column (configured at load time in :func:`load_local_llm`),
    which is what causal-LM batched generation needs to make RoPE +
    KV-cache work correctly across sequences of different lengths.
    Variable-length output is fine: ``model.generate`` natively
    short-circuits sequences that emit EOS while continuing the
    others, so the wallclock is set by the slowest sequence in the
    batch but throughput still scales close to ``B`` on H100.
    """
    import torch

    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
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
    # All rows share the same left-padded prompt length thanks to
    # ``padding=True``; slice it off uniformly.
    prompt_len = inputs.input_ids.shape[1]
    return [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
        for row in out_ids
    ]


# ---------------------------------------------------------------------------
# Progress logging — periodic ETA + batch-rate snapshot
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format a wallclock duration as ``HhMMmSSs`` / ``MMmSSs`` / ``Ns``.

    Picked over ``datetime.timedelta`` so the units land where humans
    expect on multi-hour ETAs and the string fits a log line (no leading
    "0:" prefixes, no microseconds). Negative inputs and NaN clamp to
    ``"0s"`` to keep the progress line resilient to clock skew or a
    zero-batch divisor.
    """
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN check
        return "0s"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# OOM-aware batch fallback
# ---------------------------------------------------------------------------


def _is_oom(exc: BaseException) -> bool:
    """Best-effort detection of CUDA OOM across torch versions.

    Covers ``torch.cuda.OutOfMemoryError`` (the modern subclass) and
    legacy ``RuntimeError("CUDA out of memory ...")`` strings. Used
    by :func:`generate_queries_local` to fall back to single-job
    generation for one bad batch instead of writing off ``B`` jobs
    as api errors.
    """
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda oom" in msg:
        return True
    name = type(exc).__name__
    return name == "OutOfMemoryError"


def _empty_cuda_cache_silent() -> None:
    """``torch.cuda.empty_cache()`` if available; no-op for tests / CPU.

    Called between an OOMed batch and its size-1 retry so freed KV
    cache from the failed allocation actually returns to the pool.
    Guarded with a broad except so unit tests (no torch import in
    the stub paths) are unaffected.
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — defensive in tests + non-CUDA envs
        pass


# ---------------------------------------------------------------------------
# Per-job generation — single-job + batched paths share the same helpers
# ---------------------------------------------------------------------------


def _build_messages(job: Job) -> list[dict[str, str]]:
    """Per-job system + user message pair. Shared by single + batched paths."""
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                job.query_type, job.record.lg, job.sample_idx
            ),
        },
        {
            "role": "user",
            "content": build_user_message(
                job.record, job.bucket_label, job.bucket_text, job.sample_idx
            ),
        },
    ]


def _build_query_from_parsed(
    parsed: QueryOutput, job: Job, cfg: GenerationConfig
) -> JobResult:
    """Verify references against the source bucket and assemble a :class:`Query`.

    The parse step is the caller's responsibility — this only owns
    the post-parse half (verbatim-anchor verification + Query
    construction). Shared by single-job and batched paths so both
    emit the same record shape.
    """
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


def generate_one_local(
    job: Job,
    cfg: GenerationConfig,
    *,
    generate_fn: Callable[[list[dict[str, str]]], str],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> JobResult:
    """Generate one query for ``job`` via a local model.

    ``generate_fn`` is single-message (``messages → str``); injected
    so this function can be unit-tested without loading real
    weights. Used by the batched run loop's parse-retry fallback —
    when one completion in a batch fails to parse, we retry that
    single job rather than re-running the whole batch.

    Retries on parse failure (model returned non-JSON or didn't match
    the :class:`QueryOutput` schema). Real generation errors propagate
    out of ``generate_fn`` and are tagged ``error_kind="api"`` to
    mirror the AIaaS path's failure taxonomy.
    """
    messages = _build_messages(job)
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
    return _build_query_from_parsed(parsed, job, cfg)


# ---------------------------------------------------------------------------
# Run loop — batched over ``batch_size`` jobs at a time
# ---------------------------------------------------------------------------


def generate_queries_local(
    records: Sequence,
    cfg: GenerationConfig,
    *,
    generate_fn: Callable[[list[list[dict[str, str]]]], list[str]],
    batch_size: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    log_every_n: int = DEFAULT_PROGRESS_LOG_EVERY_N_BATCHES,
) -> tuple[list[Query], GenerationStats]:
    """Drive generation across all (doc, bucket, query_type) jobs.

    ``generate_fn`` is **batched**: it takes a list of ``B`` message
    sequences and returns a list of ``B`` reply strings. With
    ``batch_size=1`` the function still runs per-job (the batch is
    a length-1 list), so single-stream behaviour is preserved
    byte-for-byte for laptop / parity-with-AIaaS runs.

    Parse-retry semantics in batched mode: a per-row parse failure
    triggers a *single-job* retry of just that prompt (not the whole
    batch), reusing :func:`generate_completion` (single-message
    surface) so the retry is one prompt, not ``B``. Up to
    ``max_retries`` total attempts per job.

    Progress: every ``log_every_n`` batches an INFO line lands in
    the per-run log file with elapsed wallclock since the first
    ``generate_fn`` call, the rolling average batch wallclock, and
    a linear-extrapolation ETA. The terminal ``tqdm`` bar still
    shows per-job progress; the periodic log line is the
    operator's view from outside the pod (``runai logs``,
    ``rcp-scratch/<user>/.../<study>/query-generate.log``).

    No semaphore / asyncio: one GPU, one inflight ``model.generate``
    call at a time. The throughput lever here is ``batch_size`` (more
    sequences per generate), not concurrency (multiple generates in
    flight), because the GPU is the bottleneck.
    """
    stats = GenerationStats(corpus_records=len(records))
    jobs = _plan_jobs(records, cfg)
    if not jobs:
        return [], stats

    bs = max(1, int(batch_size))
    total_batches = (len(jobs) + bs - 1) // bs

    # Sort jobs by descending prompt length so each batch packs
    # similar-length prompts and minimises left-padding waste during
    # prefill. The prompt is dominated by FULL ARTICLE + bucket
    # excerpt, so doc-char length is a close-enough proxy for the
    # tokenised prompt length without paying tokenizer cost upfront.
    # Stable sort: jobs with the same length keep their _plan_jobs
    # order, so single-record corpora (the unit-test shape) are
    # unaffected. Real runs land queries in length-descending order
    # — downstream eval keys on query_id so insertion order is not
    # part of the contract. With B=4 and study-A-fit's 4-8k-token
    # docs this typically saves ~10-20% on prefill wallclock; the
    # win scales with the variance of doc length within the corpus.
    jobs = sorted(jobs, key=lambda j: -len(j.record.ft))
    log.info(
        "planned: corpus_records=%d jobs=%d batches=%d batch_size=%d buckets=%s "
        "queries_per_bucket=%d model=%s",
        stats.corpus_records,
        len(jobs),
        total_batches,
        bs,
        list(cfg.position_buckets),
        cfg.queries_per_bucket,
        cfg.model,
    )

    # Single-prompt retry helper: reuses generate_fn so test stubs
    # don't need a separate hook for the retry path.
    def _single_generate(messages: list[dict[str, str]]) -> str:
        return generate_fn([messages])[0]

    queries: list[Query] = []
    pbar = tqdm(total=len(jobs), file=sys.stderr, disable=None, desc="generate", unit="q")
    # Wallclock starts at the FIRST generate_fn call, not at function
    # entry — model load + corpus read shouldn't pollute the
    # batch-throughput measurement. Using time.monotonic so a system
    # clock adjustment mid-run can't make ETAs go negative.
    start_time = time.monotonic()
    batch_count = 0
    try:
        for i in range(0, len(jobs), bs):
            batch = jobs[i : i + bs]
            messages_batch = [_build_messages(j) for j in batch]
            try:
                completions = generate_fn(messages_batch)
            except Exception as exc:  # noqa: BLE001 — whole-batch failure
                # OOM-aware fallback: one pathological long-prompt batch
                # shouldn't lose B jobs. Retry the same prompts one at a
                # time after emptying the CUDA cache; batch=1 with the
                # weights already loaded fits comfortably (model + a
                # single ~13 k-token context ≈ 62 GB on a 80 GB H100).
                # If a single-job retry itself fails (parse error, real
                # generate error), it gets tagged "api" and the next
                # prompt continues.
                if _is_oom(exc) and len(batch) > 1:
                    log.warning(
                        "OOM at batch=%d; falling back to single-job for these "
                        "%d prompts (consider lowering --batch-size for the "
                        "next run)",
                        len(batch),
                        len(batch),
                    )
                    _empty_cuda_cache_silent()
                    completions = []
                    for messages in messages_batch:
                        try:
                            completions.append(generate_fn([messages])[0])
                        except Exception as inner:  # noqa: BLE001
                            log.warning(
                                "single-job fallback also failed for one "
                                "prompt: %s",
                                inner,
                            )
                            # Empty string forces parse_query_output to
                            # return None, so the job is tagged "api"
                            # by the existing post-batch loop.
                            completions.append("")
                else:
                    log.warning(
                        "batched generate error (jobs=%d): %s", len(batch), exc
                    )
                    for _ in batch:
                        stats.attempts += 1
                        stats.api_errors += 1
                    pbar.update(len(batch))
                    continue
            if len(completions) != len(batch):
                # Defensive: the contract is one completion per job.
                log.warning(
                    "batched generate returned %d completions for %d jobs; "
                    "marking the whole batch as api errors",
                    len(completions),
                    len(batch),
                )
                for _ in batch:
                    stats.attempts += 1
                    stats.api_errors += 1
                pbar.update(len(batch))
                continue

            for job, raw in zip(batch, completions, strict=True):
                stats.attempts += 1
                parsed = parse_query_output(raw)
                # First attempt was the batched call; the parse-retry
                # loop below burns the remaining (max_retries - 1)
                # attempts as single-prompt retries. Keeps batch
                # throughput on the happy path and only pays the
                # single-job cost when the model emitted non-JSON.
                attempts_used = 1
                while parsed is None and attempts_used < max_retries:
                    log.debug(
                        "parse failed (attempt %d/%d) for ci_id=%s; "
                        "retrying single-job; raw[:200]=%r",
                        attempts_used,
                        max_retries,
                        job.record.ci_id,
                        raw[:200],
                    )
                    try:
                        raw = _single_generate(_build_messages(job))
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "single-job retry generate error for ci_id=%s: %s",
                            job.record.ci_id,
                            exc,
                        )
                        parsed = None
                        break
                    parsed = parse_query_output(raw)
                    attempts_used += 1

                if parsed is None:
                    stats.api_errors += 1
                    pbar.update(1)
                    continue

                result = _build_query_from_parsed(parsed, job, cfg)
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
                else:
                    # error_kind == "no_refs" by construction
                    stats.no_refs_returned += 1
                pbar.update(1)

            # End-of-batch progress checkpoint. Counts every batch
            # iteration of the outer loop, including OOM-fallback
            # ones (which take longer because they ran B size-1
            # generates) and whole-batch failures — the operator
            # cares about wallclock, not happy-path-only throughput.
            batch_count += 1
            if log_every_n > 0 and batch_count % log_every_n == 0:
                elapsed = time.monotonic() - start_time
                avg_batch_s = elapsed / batch_count
                remaining_batches = max(0, total_batches - batch_count)
                eta_s = avg_batch_s * remaining_batches
                log.info(
                    "progress: batches=%d/%d (%.1f%%) elapsed=%s "
                    "avg_batch=%.1fs eta=%s queries_kept=%d "
                    "api_errors=%d no_refs=%d",
                    batch_count,
                    total_batches,
                    100.0 * batch_count / total_batches,
                    _fmt_duration(elapsed),
                    avg_batch_s,
                    _fmt_duration(eta_s),
                    stats.queries_kept,
                    stats.api_errors,
                    stats.no_refs_returned,
                )
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
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "number of prompts processed per model.generate call. Default "
            "1 (single-stream, laptop-safe). The Run:AI submit target "
            "overrides to 8 on H100 80GB; bump higher only if VRAM "
            "headroom permits (Qwen3-30B-A3B bf16 ≈ 60 GB weights + "
            "~1 GB/sequence KV cache at 10k context)."
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
    every field the prompts + planner read, and these four fields are
    the only ones unique to the CaaS path.
    """

    dtype: str = DEFAULT_DTYPE
    attention: str = DEFAULT_ATTENTION
    max_retries: int = DEFAULT_MAX_RETRIES
    batch_size: int = DEFAULT_BATCH_SIZE


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
        batch_size=int(args.batch_size),
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

    def generate_fn(messages_batch: list[list[dict[str, str]]]) -> list[str]:
        return generate_completions_batched(
            model,
            tokenizer,
            messages_batch,
            max_new_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    queries, stats = generate_queries_local(
        records,
        cfg,
        generate_fn=generate_fn,
        batch_size=knobs.batch_size,
        max_retries=knobs.max_retries,
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
        "batch_size=%d corpus=s3://%s/%s output=s3://%s/%s upload=%s",
        cfg.study_name or "(none)",
        cfg.model,
        knobs.dtype,
        knobs.attention,
        knobs.batch_size,
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
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DTYPE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_PROGRESS_LOG_EVERY_N_BATCHES",
    "LOCAL_ENDPOINT",
    "LocalKnobs",
    "build_parser",
    "config_from_args",
    "generate_completion",
    "generate_completions_batched",
    "generate_one_local",
    "generate_queries_local",
    "load_local_llm",
    "main",
    "parse_query_output",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
