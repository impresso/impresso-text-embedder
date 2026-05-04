"""Run one chunking-eval scenario over the research corpus shard.

Reads the single ``.jsonl.bz2`` corpus shard produced by
:mod:`research.corpus_fetch`, embeds every record under one
:class:`research.scenarios.Scenario`, and writes a single output
shard back to S3 at ``<study_s3_root>/<scenario_id>.jsonl.bz2``
(e.g. ``chunking-eval/A-fit/S3.jsonl.bz2``). The flat per-scenario
filename means ``aws s3 ls`` of the study prefix shows the whole
sweep matrix at a glance.

Each Run:AI job runs **one scenario**; the Makefile target
``runai-submit-research-all`` loops the registry derived from the
study YAML and submits them concurrently.

The CLI is config-driven via a required ``--config <path>`` to a
study YAML (schema in :mod:`research.study_config`); the registry of
scenarios is then built by :func:`research.scenario_builder.build_scenarios`.
The S3-path CLI flags are gone — the study config is the only source
of truth for "where things live". ``--no-upload`` stays as the only
S3-side iteration knob; embed-knob overrides (``--batch-size``,
``--precision``, ``--limit``…) remain as one-shot escape hatches.

Output schema = production text-level schema
``{ci_id, model_id, embedding, size, ts, ci_type}`` *plus* research-only
metadata fields ``{lg, year, provider, alias, ocrqa, len_chars,
n_chunks, n_tokens, n_tokens_per_chunk, scenario_id, chunker,
chunk_tokens, study_name, study_config_sha}``. ``n_chunks`` and
``n_tokens`` are the load-bearing per-doc metrics for the eval —
chunked vs one-shot, and exact token length per document for
stratified analysis. ``n_tokens_per_chunk`` is a list of length
``n_chunks`` reporting the exact token count fed to the encoder for
each chunk (re-tokenised after chunker decode for ``fixed-window``).
``study_name`` + ``study_config_sha`` track which YAML config
produced the artefact.

Design and rejected alternatives in
``.progress/embedding-sweep/notes.md`` and
``.progress/study-config/notes.md``.
"""

from __future__ import annotations

import argparse
import bz2
import dataclasses
import getpass
import logging
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

from impresso_text_embedder import logging_setup
from impresso_text_embedder.embed import (
    EncoderConfig,
    LongDocConfig,
    build_embedder_tag,
    embed_records,
    is_long_doc,
)
from impresso_text_embedder.logging_setup import configure_logging
from impresso_text_embedder.research._io import staged_input, staged_output
from impresso_text_embedder.research.scenario_builder import ScenarioRegistry
from impresso_text_embedder.research.scenarios import Scenario
from impresso_text_embedder.research.study_config import (
    CORPUS_FILENAME,
    StudyConfig,
    load_study_config,
    scenario_filename,
)

log = logging.getLogger(__name__)


# Research metadata fields preserved from the corpus shard. Mirrors the
# subset of corpus_fetch's output that is useful for stratified eval.
_METADATA_FIELDS: tuple[str, ...] = (
    "lg",
    "year",
    "provider",
    "alias",
    "ocrqa",
    "len_chars",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-research-embed-sweep",
        description=(
            "Embed the chunking-eval corpus under one scenario from the "
            "registry derived from a study YAML; one runai job per scenario."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        help=(
            "Study YAML config (e.g. configs/research/study-v1.yaml). "
            "Required unless --list-table is the only thing requested."
        ),
    )
    p.add_argument(
        "--scenario",
        help="scenario id from the registry (e.g. S3). Use --list to see all.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="print the scenario table for the given --config and exit",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "skip the S3 upload; the output lands at the study's local "
            "mirror path (paths.local_root + scenario filename) so the "
            "user can find it deterministically."
        ),
    )
    # Embedder overrides (default = from --config).
    p.add_argument("--model-name", default=None)
    p.add_argument("--model-revision", default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="encode batch size; per-GPU profile default if omitted",
    )
    p.add_argument(
        "--min-char-length",
        type=int,
        default=None,
        help="minimum reconstructed-text length to keep a record",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most the first N corpus records (smoke tests)",
    )
    p.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default=None,
        help="encode-time precision; same flag as the production CLI",
    )
    p.add_argument(
        "--attention",
        choices=["xformers", "eager"],
        default=None,
    )
    p.add_argument(
        "--unpad-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--log-level-file",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="level for the per-scenario log file (terminal stays ERROR-only)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "override the base log directory; the full path becomes "
            "<log-dir>/<study>/<YYYY-MM-DD>/<scenario_id>.log. Default is "
            "/rcp-scratch/<user>/experiments/chunking-eval (requires the "
            "PVC mounted). Outside RCP, --log-dir is required."
        ),
    )
    return p


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in dependencies
        return
    load_dotenv()


def _resolve_research_log_dir(log_dir: Path | None) -> Path:
    """Pick the base directory for sweep log files.

    Mirrors :func:`logging_setup._resolve_log_path` semantics but lands
    research logs under ``…/experiments/chunking-eval`` instead of
    ``…/experiments/embeddings`` so they don't mix with production-CLI
    runs. Fail-fast when the PVC is not mounted and no override was
    given — silent fallback to ``./rcp-scratch/...`` would mask a
    misconfigured RCP pod.
    """
    if log_dir is not None:
        return log_dir
    if not logging_setup.DEFAULT_RCP_SCRATCH.is_dir():
        raise SystemExit(
            f"Default log path requires {logging_setup.DEFAULT_RCP_SCRATCH} "
            "to be mounted (Run:AI PVC). Mount it, or pass --log-dir <path> "
            "to override."
        )
    return (
        logging_setup.DEFAULT_RCP_SCRATCH
        / getpass.getuser()
        / "experiments"
        / "chunking-eval"
    )


def _build_token_counter(model: Any) -> Callable[[str], int]:
    """Return a closure that counts tokens via ``model.tokenizer``.

    ``verbose=False`` mirrors create.py: silences transformers' "sequence
    length > model_max_length" warning since we count to decide whether to
    chunk and to record metadata, never feed raw ids to the model.
    """
    tokenizer = model.tokenizer

    def _count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))

    return _count


def build_long_doc_config(scenario: Scenario, model: object) -> LongDocConfig | None:
    """Construct the :class:`LongDocConfig` for a given scenario.

    Returns ``None`` for the truncate scenario (``S0``); the encoder then
    falls through to the production tokenizer-truncates-at-max path.

    For chunked scenarios, ``model_max_tokens`` is set equal to
    ``scenario.chunk_tokens`` so the chunker fires for *every* doc longer
    than the target — that is the behaviour the research question
    requires (see :mod:`research.scenarios` module docstring).
    """
    if scenario.chunker_name is None:
        return None

    from impresso_text_embedder.aggregation import get_strategy as get_aggregator
    from impresso_text_embedder.chunking import get_strategy as get_chunker

    tokenizer = model.tokenizer  # type: ignore[attr-defined]
    chunk_tokens = scenario.chunk_tokens
    assert chunk_tokens is not None  # invariant: chunked scenarios set this

    _count = _build_token_counter(model)

    chunker_kwargs: dict[str, Any]
    if scenario.chunker_name == "fixed-window":
        chunker_kwargs = {"tokenizer": tokenizer, "max_tokens": chunk_tokens}
    elif scenario.chunker_name == "token-budget":
        chunker_kwargs = {"token_counter": _count, "max_tokens": chunk_tokens}
    elif scenario.chunker_name == "semantic":
        # chonkie's chunk_size is the soft target; the trigger
        # (model_max_tokens) is set to the same value below so semantic
        # chunking actually fires on docs at that scale. Passing
        # `tokenizer=` makes chonkie size chunks in GTE tokens instead
        # of its default `potion-base-8M` WordPiece tokens, so
        # chunk_tokens means the same thing across all three chunker
        # families. See `chunking/semantic.py` module docstring.
        chunker_kwargs = {"chunk_size": chunk_tokens, "tokenizer": tokenizer}
    else:
        raise ValueError(
            f"scenario {scenario.id}: unknown chunker {scenario.chunker_name!r}"
        )

    chunker = get_chunker(scenario.chunker_name, **chunker_kwargs)
    aggregator_name = scenario.aggregator_name
    assert aggregator_name is not None
    aggregator = get_aggregator(aggregator_name)
    return LongDocConfig(
        strategy="chunk",
        chunker=chunker,
        aggregator=aggregator,
        model_max_tokens=chunk_tokens,
        token_counter=_count,
    )


def _read_corpus(local_path: Path) -> list[dict]:
    """Load every record from the corpus shard into memory."""
    records: list[dict] = []
    with bz2.open(local_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(orjson.loads(line))
    return records


@dataclasses.dataclass(frozen=True)
class ChunkStats:
    """Per-record token + chunk metadata recorded alongside the embedding.

    ``n_tokens`` is the exact token length of the original document
    (single ``tokenizer.encode`` pass on ``ft``), stable across all 16
    scenarios for a given ``ci_id``. ``n_tokens_per_chunk`` is the
    per-chunk token count after the chunker has run — for ``fixed-window``
    these are re-tokenisations of the decoded chunk strings, so the sum
    can drift slightly from ``n_tokens`` (a known L1 redundancy in
    :mod:`chunking.fixed_window`); for ``token-budget`` and ``semantic``
    they are exact counts of the raw text slices.
    """

    n_chunks: int
    n_tokens: int
    n_tokens_per_chunk: list[int]

    @classmethod
    def empty(cls) -> ChunkStats:
        return cls(n_chunks=0, n_tokens=0, n_tokens_per_chunk=[])

    @classmethod
    def one_shot(cls, n_tokens: int) -> ChunkStats:
        return cls(n_chunks=1, n_tokens=n_tokens, n_tokens_per_chunk=[n_tokens])


def _compute_chunk_stats(
    records: Iterable[dict],
    token_counter: Callable[[str], int],
    long_doc: LongDocConfig | None,
) -> dict[str, ChunkStats]:
    """Pre-compute :class:`ChunkStats` per ``ci_id`` for the eval metadata.

    Always tokenises the original document once for ``n_tokens`` (so the
    field is populated even on the truncate baseline ``S0``). For chunked
    scenarios where the doc actually crosses the threshold, also tokenises
    each chunk to populate ``n_tokens_per_chunk``.
    """
    out: dict[str, ChunkStats] = {}
    chunker = (
        long_doc.chunker if (long_doc is not None and long_doc.is_active()) else None
    )
    for rec in records:
        ci_id = rec.get("ci_id") or rec.get("id")
        if not ci_id:
            continue
        ci_id = str(ci_id)
        text = rec.get("ft") or ""
        if not text:
            out[ci_id] = ChunkStats.empty()
            continue
        n_tokens = token_counter(text)
        if chunker is None or not is_long_doc(text, long_doc):
            out[ci_id] = ChunkStats.one_shot(n_tokens)
            continue
        chunks = chunker.chunk(text)
        per_chunk = [token_counter(c.text) for c in chunks if c.text]
        if not per_chunk:
            # All-empty chunker output → fall back to one-shot, matching the
            # prior max(n, 1) defensive behaviour.
            out[ci_id] = ChunkStats.one_shot(n_tokens)
            continue
        out[ci_id] = ChunkStats(
            n_chunks=len(per_chunk),
            n_tokens=n_tokens,
            n_tokens_per_chunk=per_chunk,
        )
    return out


def _to_input_record(corpus_rec: dict) -> dict:
    """Adapt a corpus record to the shape :func:`embed_records` expects."""
    return {
        "id": corpus_rec.get("ci_id"),
        "tp": corpus_rec.get("tp"),
        "ft": corpus_rec.get("ft"),
        "sents": corpus_rec.get("sents"),
    }


def _build_output_record(
    embed_out: dict,
    corpus_rec: dict,
    chunk_stats: ChunkStats,
    scenario: Scenario,
    study_name: str | None = None,
    study_config_sha: str | None = None,
) -> dict:
    """Merge the production embed output with research metadata."""
    enriched = dict(embed_out)
    for field in _METADATA_FIELDS:
        if field in corpus_rec:
            enriched[field] = corpus_rec[field]
    enriched["n_chunks"] = chunk_stats.n_chunks
    enriched["n_tokens"] = chunk_stats.n_tokens
    enriched["n_tokens_per_chunk"] = chunk_stats.n_tokens_per_chunk
    enriched["scenario_id"] = scenario.id
    enriched["chunker"] = scenario.chunker_name or "truncate"
    enriched["chunk_tokens"] = scenario.chunk_tokens
    enriched["aggregator"] = scenario.aggregator_name or "(none)"
    if study_name is not None:
        enriched["study_name"] = study_name
    if study_config_sha is not None:
        enriched["study_config_sha"] = study_config_sha
    return enriched


@dataclasses.dataclass
class SweepStats:
    """Counters reported per run."""

    corpus_records: int = 0
    embedded: int = 0
    filter_counter: Counter[str] = dataclasses.field(default_factory=Counter)
    n_chunks_distribution: Counter[int] = dataclasses.field(default_factory=Counter)


def run_sweep(
    scenario: Scenario,
    corpus_records: list[dict],
    model: Any,
    encoder_cfg: EncoderConfig,
    embedder_tag: str,
    local_output: Path,
    study_name: str | None = None,
    study_config_sha: str | None = None,
) -> SweepStats:
    """Embed ``corpus_records`` under ``scenario`` and write to ``local_output``."""
    stats = SweepStats(corpus_records=len(corpus_records))
    by_ci_id = {str(r.get("ci_id") or r.get("id")): r for r in corpus_records}

    token_counter = _build_token_counter(model)
    chunk_stats = _compute_chunk_stats(
        corpus_records, token_counter, encoder_cfg.long_doc
    )
    for cs in chunk_stats.values():
        stats.n_chunks_distribution[cs.n_chunks] += 1

    input_records = (_to_input_record(r) for r in corpus_records)
    out_iter = embed_records(
        records=input_records,
        level="text",
        model=model,
        cfg=encoder_cfg,
        embedder_tag=embedder_tag,
        filter_counter=stats.filter_counter,
    )

    local_output.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(local_output, "wb") as fh:
        for embed_out in tqdm(
            out_iter,
            total=len(corpus_records),
            file=sys.stderr,
            disable=None,
            desc=scenario.id,
            unit="rec",
        ):
            ci_id = str(embed_out.get("ci_id"))
            corpus_rec = by_ci_id.get(ci_id, {})
            cs = chunk_stats.get(ci_id, ChunkStats.one_shot(0))
            enriched = _build_output_record(
                embed_out,
                corpus_rec,
                cs,
                scenario,
                study_name=study_name,
                study_config_sha=study_config_sha,
            )
            fh.write(orjson.dumps(enriched, option=orjson.OPT_APPEND_NEWLINE))
            stats.embedded += 1
    return stats


def _format_stats(stats: SweepStats) -> str:
    parts = [
        f"corpus_records={stats.corpus_records}",
        f"embedded={stats.embedded}",
    ]
    if stats.filter_counter:
        filters = " ".join(f"{k}={v}" for k, v in sorted(stats.filter_counter.items()))
        parts.append(f"filtered=({filters})")
    if stats.n_chunks_distribution:
        # Buckets the n_chunks distribution into 1 / 2-4 / 5-16 / 17+.
        buckets = {"=1": 0, "2-4": 0, "5-16": 0, "17+": 0}
        for n, count in stats.n_chunks_distribution.items():
            if n <= 1:
                buckets["=1"] += count
            elif n <= 4:
                buckets["2-4"] += count
            elif n <= 16:
                buckets["5-16"] += count
            else:
                buckets["17+"] += count
        parts.append(
            "n_chunks=(" + " ".join(f"{k}={v}" for k, v in buckets.items()) + ")"
        )
    return " ".join(parts)


def _resolve_value(cli: Any, cfg_val: Any, fallback: Any) -> Any:
    """CLI > config > fallback. ``None`` means "not provided"."""
    if cli is not None:
        return cli
    if cfg_val is not None:
        return cfg_val
    return fallback


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.config:
        parser.error("--config is required (study YAML, e.g. configs/research/study-v1.yaml)")

    study_cfg: StudyConfig = load_study_config(args.config)
    registry = ScenarioRegistry.from_study(study_cfg)

    if args.list:
        print(registry.format_table())
        return 0

    if not args.scenario:
        parser.error("--scenario is required (use --list to see ids)")

    try:
        scenario = registry.get(args.scenario)
    except KeyError as exc:
        parser.error(str(exc))
        return 2  # unreachable; argparse.error exits

    log_base = _resolve_research_log_dir(args.log_dir) / study_cfg.study.name
    log_path = configure_logging(
        provider=scenario.id,
        log_dir=log_base,
        log_level_file=args.log_level_file,
    )
    print(f"logging to {log_path}", file=sys.stderr)

    _load_env()

    # Embed knobs: CLI > YAML > fallback. Path knobs are NOT here —
    # the study config is the only source of truth for "where things live".
    model_name = _resolve_value(args.model_name, study_cfg.embed.model_name, None)
    model_revision = _resolve_value(
        args.model_revision, study_cfg.embed.model_revision, None
    )
    precision = _resolve_value(args.precision, study_cfg.embed.precision, "bf16")
    attention = _resolve_value(args.attention, study_cfg.embed.attention, "xformers")
    unpad_inputs = _resolve_value(
        args.unpad_inputs, study_cfg.embed.unpad_inputs, True
    )
    min_char_length = _resolve_value(
        args.min_char_length, study_cfg.embed.min_char_length, 0
    )

    if args.batch_size is None:
        from impresso_text_embedder.accel import detect_profile

        batch_size = detect_profile().default_batch_size
    else:
        batch_size = args.batch_size

    bucket = study_cfg.s3.bucket
    corpus_key = study_cfg.s3_key(CORPUS_FILENAME)
    out_filename = scenario_filename(scenario.id)
    out_key = study_cfg.s3_key(out_filename)
    out_local_mirror = study_cfg.local_path(out_filename)

    log.info(
        "study=%s study_sha=%s scenario=%s chunker=%s chunk_tokens=%s aggregator=%s "
        "model=%s@%s batch_size=%d precision=%s",
        study_cfg.study.name,
        study_cfg.config_sha,
        scenario.id,
        scenario.chunker_name or "(truncate)",
        scenario.chunk_tokens,
        scenario.aggregator_name or "(none)",
        model_name,
        model_revision,
        batch_size,
        precision,
    )
    log.info(
        "corpus=s3://%s/%s output=s3://%s/%s upload=%s",
        bucket,
        corpus_key,
        bucket,
        out_key,
        "no" if args.no_upload else "yes",
    )

    from impresso_text_embedder.model import load_model

    model = load_model(
        name=model_name,
        revision=model_revision,
        use_xformers=attention == "xformers",
        unpad_inputs=unpad_inputs,
    )

    long_doc = build_long_doc_config(scenario, model)
    encoder_cfg = EncoderConfig(
        batch_size=batch_size,
        min_char_length=min_char_length,
        precision=precision,
        long_doc=long_doc,
    )
    embedder_tag = build_embedder_tag(model_name, model_revision)

    # Read the corpus into memory then release the staged input — the
    # output write doesn't need the input file on disk.
    with staged_input(bucket, corpus_key) as corpus_path:
        records = _read_corpus(corpus_path)
    if args.limit is not None:
        records = records[: args.limit]

    with staged_output(
        bucket, out_key, out_local_mirror, upload=not args.no_upload
    ) as out_path:
        stats = run_sweep(
            scenario=scenario,
            corpus_records=records,
            model=model,
            encoder_cfg=encoder_cfg,
            embedder_tag=embedder_tag,
            local_output=out_path,
            study_name=study_cfg.study.name,
            study_config_sha=study_cfg.config_sha,
        )
    log.info("sweep stats: %s", _format_stats(stats))

    return 0


__all__ = ["ChunkStats", "build_long_doc_config", "main", "run_sweep"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
