"""Seed a new study's S3 prefix with artefacts copied from an existing study.

When two studies share their corpus filters and query budget — same
article selection, same generated queries, same query embeddings —
the only thing that legitimately differs is the scenario grid. Rather
than re-running ``corpus_select`` / ``corpus_fetch`` /
``query_generate`` / ``query_embed`` against a YAML that produces the
identical artefacts, this CLI server-side-copies the corpus, queries,
and queries-embedded shards from the source study's S3 prefix to the
target study's prefix.

Server-side via ``s3.copy_object`` (see :func:`io.copy_s3_object`):
no body transfer, no MD5 recompute, idempotent across reruns. The
copy lands at the destination keys ``cfg.s3_key(...)`` resolves to,
so the downstream CLIs (``embed_sweep`` etc.) find the artefacts at
the same conventional locations they would have otherwise produced.

Cross-bucket copies are out of scope (would need extra IAM thought);
both studies must live in the same ``s3.bucket``. Identical source
and target study names are rejected — that would be a same-key copy
and almost certainly a mistake.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from impresso_text_embedder.io import copy_s3_object
from impresso_text_embedder.research.study_config import (
    CORPUS_FILENAME,
    QUERIES_EMBEDDED_FILENAME,
    QUERIES_FILENAME,
    StudyConfig,
    load_study_config,
)

log = logging.getLogger(__name__)


# Shorthand keys (CLI-friendly) → conventional filenames under the
# study's S3 prefix. Defined here, not in study_config, because this
# CLI is the only consumer of the mapping.
_ARTIFACT_FILENAMES: dict[str, str] = {
    "corpus": CORPUS_FILENAME,
    "queries": QUERIES_FILENAME,
    "queries-embedded": QUERIES_EMBEDDED_FILENAME,
}
_DEFAULT_ARTIFACTS: tuple[str, ...] = (
    "corpus",
    "queries",
    "queries-embedded",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-research-study-seed",
        description=(
            "Server-side-copy corpus / queries / queries-embedded from "
            "one study's S3 prefix to another. Skips destination keys "
            "that already exist unless --overwrite is set; --dry-run "
            "prints the src->dst pairs without touching S3."
        ),
    )
    p.add_argument(
        "--source-config",
        required=True,
        type=Path,
        help="Source study YAML (the study whose artefacts get copied).",
    )
    p.add_argument(
        "--target-config",
        required=True,
        type=Path,
        help="Target study YAML (the study whose S3 prefix gets seeded).",
    )
    p.add_argument(
        "--artifacts",
        type=str,
        default=",".join(_DEFAULT_ARTIFACTS),
        help=(
            "Comma-separated artefact names to copy. Choices: "
            f"{', '.join(_ARTIFACT_FILENAMES)}. Default: all three."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination objects that already exist (default: skip).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the src->dst pairs without calling S3.",
    )
    return p


def _parse_artifacts(arg: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in arg.split(",") if name.strip())
    if not names:
        raise SystemExit("--artifacts must list at least one name")
    bad = [n for n in names if n not in _ARTIFACT_FILENAMES]
    if bad:
        raise SystemExit(
            f"unknown artifact name(s) {bad}; "
            f"choices: {sorted(_ARTIFACT_FILENAMES)}"
        )
    return names


def _resolve_pairs(
    src_cfg: StudyConfig,
    dst_cfg: StudyConfig,
    artifacts: Sequence[str],
) -> list[tuple[str, str, str, str]]:
    """Return ``[(src_bucket, src_key, dst_bucket, dst_key), ...]``."""
    pairs: list[tuple[str, str, str, str]] = []
    for name in artifacts:
        filename = _ARTIFACT_FILENAMES[name]
        src_key = src_cfg.s3_key(filename)
        dst_key = dst_cfg.s3_key(filename)
        pairs.append((src_cfg.s3.bucket, src_key, dst_cfg.s3.bucket, dst_key))
    return pairs


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in dependencies
        return
    load_dotenv()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    src_cfg = load_study_config(args.source_config)
    dst_cfg = load_study_config(args.target_config)
    if src_cfg.study.name == dst_cfg.study.name:
        raise SystemExit(
            f"source and target study names are identical ({src_cfg.study.name!r}); "
            "seeding a study from itself would copy keys onto themselves."
        )
    if src_cfg.s3.bucket != dst_cfg.s3.bucket:
        raise SystemExit(
            f"cross-bucket copies not supported "
            f"(src bucket={src_cfg.s3.bucket!r}, dst bucket={dst_cfg.s3.bucket!r})."
        )

    artifacts = _parse_artifacts(args.artifacts)
    pairs = _resolve_pairs(src_cfg, dst_cfg, artifacts)

    if args.dry_run:
        print(
            f"# dry-run: {len(pairs)} artefact(s), "
            f"{src_cfg.study.name!r} -> {dst_cfg.study.name!r}"
        )
        for src_bucket, src_key, dst_bucket, dst_key in pairs:
            print(f"s3://{src_bucket}/{src_key} -> s3://{dst_bucket}/{dst_key}")
        return 0

    _load_env()
    n_copied = 0
    n_skipped = 0
    for src_bucket, src_key, dst_bucket, dst_key in pairs:
        copied = copy_s3_object(
            src_bucket, src_key, dst_bucket, dst_key, overwrite=args.overwrite
        )
        if copied:
            n_copied += 1
        else:
            n_skipped += 1
    log.info(
        "seed done: %d copied, %d skipped (study %s -> %s)",
        n_copied,
        n_skipped,
        src_cfg.study.name,
        dst_cfg.study.name,
    )
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
