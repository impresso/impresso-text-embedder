"""Embed every query in ``<study>/queries.jsonl.bz2`` once.

Step 7 of ``research/chunking-eval``: produce a per-query embedding
sidecar ``<study>/queries-embedded.jsonl.bz2`` so the eval step
inside step 6 can join against per-scenario doc embeddings without
re-loading the embedding model. Mirrors :mod:`research.embed_sweep`
scaffolding (study YAML, ``staged_input``/``staged_output``,
log-dir convention, ``--no-upload`` / ``--limit`` knobs,
``_resolve_value(cli, cfg, fallback)`` precedence) and reuses the
same pinned model declared in the study YAML's ``embed:`` block.

Differs from :mod:`research.embed_sweep` in three ways: input is
the queries shard (not the corpus); chunking, aggregation, and
record filtering are skipped (queries are short plain strings —
never long-doc, no ``tp``, no ``sents``); each input row produces
exactly one output row (no scenario fan-out).

Per-record output schema combines the production text-level shape
(``{ci_id, model_id, embedding, size, ts}``) with the query identity
(``query_id``) and the eval-relevant query metadata mirrored
verbatim from the queries shard (``{lg, query_type,
position_bucket, position_chars, references, query_text}``) plus
study provenance (``{study_name, study_config_sha}``). The eval
reads one file per query, joins on ``ci_id`` against the doc
embedding shards from step 3, and never has to read
``queries.jsonl.bz2`` directly.

Design and rejected alternatives in
``.progress/query-embed/notes.md``.
"""

from __future__ import annotations

import argparse
import bz2
import dataclasses
import getpass
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

from impresso_text_embedder import logging_setup
from impresso_text_embedder.embed import build_embedder_tag
from impresso_text_embedder.logging_setup import configure_logging
from impresso_text_embedder.model import encode_texts
from impresso_text_embedder.research._io import staged_input, staged_output
from impresso_text_embedder.research.study_config import (
    QUERIES_EMBEDDED_FILENAME,
    QUERIES_FILENAME,
    StudyConfig,
    load_study_config,
)
from impresso_text_embedder.schema import EMBEDDING_DECIMALS, utc_timestamp

log = logging.getLogger(__name__)


# Fields preserved verbatim from the queries shard onto the embedded
# shard so the eval can stratify in one pass without re-joining
# queries.jsonl.bz2. Rationale + rejected alternatives in
# .progress/query-embed/notes.md.
_MIRRORED_FIELDS: tuple[str, ...] = (
    "lg",
    "query_type",
    "position_bucket",
    "position_chars",
    "references",
    "query_text",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-research-query-embed",
        description=(
            "Embed every query in <study>/queries.jsonl.bz2 once and "
            "write a per-query <study>/queries-embedded.jsonl.bz2 "
            "sidecar. One CLI per study; the embedder model is the "
            "one pinned in the study YAML's embed: block."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Study YAML config (e.g. configs/research/study-v1.yaml). Required.",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "skip the S3 upload; the output lands at the study's "
            "local mirror path (paths.local_root + "
            "queries-embedded.jsonl.bz2)."
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
        "--limit",
        type=int,
        default=None,
        help="encode at most the first N queries (smoke runs)",
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
        help="level for the per-run log file (terminal stays ERROR-only)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "override the base log directory; the full path becomes "
            "<log-dir>/<study>/<YYYY-MM-DD>/query-embed.log. Default "
            "is /rcp-scratch/<user>/experiments/chunking-eval "
            "(requires the PVC mounted). Outside RCP, --log-dir is "
            "required."
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
    """Pick the base directory for query-embed log files.

    Mirrors :func:`research.embed_sweep._resolve_research_log_dir`:
    PVC default unless ``--log-dir`` is set; fail-fast when the PVC
    is not mounted (silent fallback would mask a misconfigured RCP
    pod).
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


def _read_queries(local_path: Path) -> list[dict]:
    """Load every record from the queries shard into memory."""
    records: list[dict] = []
    with bz2.open(local_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(orjson.loads(line))
    return records


def _round_embedding(vec: list[float]) -> list[float]:
    return [round(float(x), EMBEDDING_DECIMALS) for x in vec]


def _build_output_record(
    query: dict,
    embedding: list[float],
    *,
    embedder_tag: str,
    ts: str,
    study_name: str | None,
    study_config_sha: str | None,
) -> dict:
    """Combine TextRecord-shape primitives with mirrored query metadata."""
    out: dict[str, Any] = {
        "query_id": query["query_id"],
        "ci_id": query["ci_id"],
        "embedding": _round_embedding(embedding),
        "size": len(embedding),
        "model_id": embedder_tag,
        "ts": ts,
    }
    for key in _MIRRORED_FIELDS:
        if key in query:
            out[key] = query[key]
    if study_name is not None:
        out["study_name"] = study_name
    if study_config_sha is not None:
        out["study_config_sha"] = study_config_sha
    return out


@dataclasses.dataclass
class EmbedStats:
    """Counters reported per run."""

    queries_total: int = 0
    embedded: int = 0
    skipped_empty: int = 0
    dim: int | None = None


def run_query_embed(
    queries: list[dict],
    *,
    model: Any,
    batch_size: int,
    precision: str,
    embedder_tag: str,
    local_output: Path,
    study_name: str | None = None,
    study_config_sha: str | None = None,
) -> EmbedStats:
    """Encode every query and write per-query records to ``local_output``.

    Filters out queries whose ``query_text`` is empty after
    ``str.strip`` (defensive — should not happen given step 4's
    verbatim-anchor verification, but cheap to skip). Calls
    :func:`encode_texts` once on the full batch — queries are short
    by construction (LLM output cap ~1500 tokens) and never trigger
    long-doc routing.
    """
    stats = EmbedStats(queries_total=len(queries))

    keepers: list[dict] = []
    for q in queries:
        text = (q.get("query_text") or "").strip()
        if not text:
            stats.skipped_empty += 1
            continue
        keepers.append(q)

    local_output.parent.mkdir(parents=True, exist_ok=True)

    if not keepers:
        # Touch the file so downstream consumers don't have to
        # special-case an empty input.
        with bz2.open(local_output, "wb"):
            pass
        return stats

    texts = [q["query_text"] for q in keepers]
    vectors = encode_texts(
        model=model,
        texts=texts,
        batch_size=batch_size,
        show_progress_bar=False,
        precision=precision,
    )
    # Cast once to plain Python so the rest of the loop is
    # numpy-agnostic; tests can patch encode_texts to return either
    # an ndarray or a plain list-of-lists.
    rows: list[list[float]] = (
        vectors.tolist() if hasattr(vectors, "tolist") else [list(v) for v in vectors]
    )
    stats.dim = len(rows[0]) if rows else None

    ts = utc_timestamp()
    with bz2.open(local_output, "wb") as fh:
        for q, vec in tqdm(
            zip(keepers, rows, strict=True),
            total=len(keepers),
            file=sys.stderr,
            disable=None,
            desc="query-embed",
            unit="q",
        ):
            rec = _build_output_record(
                q,
                vec,
                embedder_tag=embedder_tag,
                ts=ts,
                study_name=study_name,
                study_config_sha=study_config_sha,
            )
            fh.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))
            stats.embedded += 1
    return stats


def _format_stats(stats: EmbedStats, embedder_tag: str) -> str:
    parts = [
        f"queries_total={stats.queries_total}",
        f"embedded={stats.embedded}",
    ]
    if stats.skipped_empty:
        parts.append(f"skipped_empty={stats.skipped_empty}")
    if stats.dim is not None:
        parts.append(f"dim={stats.dim}")
    parts.append(f"model={embedder_tag}")
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

    log_base = _resolve_research_log_dir(args.log_dir) / study_cfg.study.name
    log_path = configure_logging(
        provider="query-embed",
        log_dir=log_base,
        log_level_file=args.log_level_file,
    )
    print(f"logging to {log_path}", file=sys.stderr)

    _load_env()

    model_name = _resolve_value(args.model_name, study_cfg.embed.model_name, None)
    model_revision = _resolve_value(
        args.model_revision, study_cfg.embed.model_revision, None
    )
    precision = _resolve_value(args.precision, study_cfg.embed.precision, "bf16")
    attention = _resolve_value(args.attention, study_cfg.embed.attention, "xformers")
    unpad_inputs = _resolve_value(
        args.unpad_inputs, study_cfg.embed.unpad_inputs, True
    )

    if args.batch_size is None:
        from impresso_text_embedder.accel import detect_profile

        batch_size = detect_profile().default_batch_size
    else:
        batch_size = args.batch_size

    bucket = study_cfg.s3.bucket
    queries_key = study_cfg.s3_key(QUERIES_FILENAME)
    out_key = study_cfg.s3_key(QUERIES_EMBEDDED_FILENAME)
    out_local_mirror = study_cfg.local_path(QUERIES_EMBEDDED_FILENAME)

    log.info(
        "study=%s study_sha=%s model=%s@%s batch_size=%d precision=%s "
        "attention=%s unpad_inputs=%s",
        study_cfg.study.name,
        study_cfg.config_sha,
        model_name,
        model_revision,
        batch_size,
        precision,
        attention,
        unpad_inputs,
    )
    log.info(
        "queries=s3://%s/%s output=s3://%s/%s upload=%s",
        bucket,
        queries_key,
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

    embedder_tag = build_embedder_tag(model_name, model_revision)

    with staged_input(bucket, queries_key) as queries_path:
        queries = _read_queries(queries_path)
    if args.limit is not None:
        queries = queries[: args.limit]

    with staged_output(
        bucket, out_key, out_local_mirror, upload=not args.no_upload
    ) as out_path:
        stats = run_query_embed(
            queries=queries,
            model=model,
            batch_size=batch_size,
            precision=precision,
            embedder_tag=embedder_tag,
            local_output=out_path,
            study_name=study_cfg.study.name,
            study_config_sha=study_cfg.config_sha,
        )
    log.info("query-embed stats: %s", _format_stats(stats, embedder_tag))

    return 0


__all__ = ["EmbedStats", "main", "run_query_embed"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
