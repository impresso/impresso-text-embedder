"""Materialise the chunking-eval corpus from a manifest into a single bz2 shard.

Iterates the per-language manifest produced by
:mod:`impresso_text_embedder.research.corpus_select`, groups its entries by
their ``(rebuilt_bucket, rebuilt_key)`` source file, streams each rebuilt
shard from S3 once via :func:`io.iter_jsonl_bz2`, picks out the records whose
``id`` is in the manifest, reconstructs ``ft`` from token offsets when the
rebuilt record does not already carry it (using
:func:`text.rebuild_ft_from_offsets`, the same helper the production pipeline
uses), and writes a single ``.jsonl.bz2`` corpus shard locally — then uploads
it to S3 at ``<study_s3_root>/corpus.jsonl.bz2`` (e.g.
``chunking-eval/A-fit/corpus.jsonl.bz2``). This is the input artefact every
downstream chunking-strategy sweep on this branch consumes.

The output is research scratch, not a production embedding output: it lands
under the sandbox bucket at the per-study research prefix, NOT under the
``embeddings/docs/<model-slug>/...`` convention, so it can never be confused
with a real embedding shard.

CLI surface is config-driven via ``--config <path>``; the manifest path,
output bucket, and output key all derive from the study YAML
(``paths.local_root`` + ``paths.s3_root``). ``--no-upload`` is the only
S3-side iteration knob: it writes the shard to the local mirror at
``<local_root>/corpus.jsonl.bz2`` instead of uploading.

Design rationale, rejected alternatives, and follow-ups in
``.progress/corpus-fetch/notes.md``.
"""

from __future__ import annotations

import argparse
import bz2
import dataclasses
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

from impresso_text_embedder import io as s3io
from impresso_text_embedder.research._io import staged_output
from impresso_text_embedder.research.study_config import (
    CORPUS_FILENAME,
    MANIFEST_FILENAME,
    load_study_config,
)
from impresso_text_embedder.text import rebuild_ft_from_offsets

log = logging.getLogger(__name__)

# 16 single-connection streaming readers in parallel. Streaming +
# early-break beats multipart-parallel-download on this workload because
# the manifest's median 1 record per yearly shard means we never need to
# read past the first matching line — multipart insists on fetching the
# whole shard before we can scan it. One connection per worker so 16
# workers ≈ 16 concurrent S3 connections, well within Ceph RadosGW's
# comfort zone.
DEFAULT_MAX_WORKERS: int = 16


@dataclasses.dataclass(frozen=True)
class ManifestEntry:
    """One row from ``corpus-manifest.jsonl``.

    Mirrors :class:`research.corpus_select.ManifestEntry` but defined locally
    so this module can be invoked against any manifest file without depending
    on the selector's import path. Fields not present in the manifest are
    tolerated (we only require the ones we actually use).
    """

    ci_id: str
    lg: str
    year: int
    len_chars: int
    ocrqa: float
    provider: str
    alias: str
    rebuilt_bucket: str
    rebuilt_key: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManifestEntry:
        return cls(
            ci_id=raw["ci_id"],
            lg=raw["lg"],
            year=int(raw["year"]),
            len_chars=int(raw["len_chars"]),
            ocrqa=float(raw["ocrqa"]),
            provider=raw["provider"],
            alias=raw["alias"],
            rebuilt_bucket=raw["rebuilt_bucket"],
            rebuilt_key=raw["rebuilt_key"],
        )


@dataclasses.dataclass
class FetchStats:
    """Counters reported per run."""

    manifest_entries: int = 0
    rebuilt_files: int = 0
    written: int = 0
    missing_in_rebuilt: list[str] = dataclasses.field(default_factory=list)
    ft_reconstructed: int = 0
    ft_from_record: int = 0


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Load every line of ``path`` (plain ``.jsonl``) into a manifest list."""
    entries: list[ManifestEntry] = []
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            entries.append(ManifestEntry.from_dict(orjson.loads(line)))
    return entries


def _group_by_source(
    entries: Iterable[ManifestEntry],
) -> dict[tuple[str, str], dict[str, ManifestEntry]]:
    """Bucket manifest entries by their rebuilt source file.

    Returns ``{(bucket, key): {ci_id: entry}}``. The inner dict is keyed by
    ``ci_id`` so the per-file streaming pass can do constant-time membership
    checks while reading the rebuilt shard.
    """
    groups: dict[tuple[str, str], dict[str, ManifestEntry]] = defaultdict(dict)
    for entry in entries:
        groups[(entry.rebuilt_bucket, entry.rebuilt_key)][entry.ci_id] = entry
    return groups


def _build_output_record(
    entry: ManifestEntry,
    rebuilt: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Combine manifest metadata with the rebuilt record's payload.

    Returns ``(record, ft_was_reconstructed)``. ``ft`` falls back to
    :func:`text.rebuild_ft_from_offsets` when the rebuilt record does not
    carry a precomputed full-text — same helper and same byte-for-byte
    semantics as the production pipeline (``text.rebuild_ft_from_offsets``).
    ``sents`` is preserved on the way through so downstream sentence-aware
    chunkers can use the existing tokenisation rather than re-splitting.
    """
    ft = rebuilt.get("ft")
    reconstructed = False
    if not ft:
        ft = rebuild_ft_from_offsets(rebuilt.get("sents", []) or [])
        reconstructed = True
    out: dict[str, Any] = {
        "ci_id": entry.ci_id,
        "lg": entry.lg,
        "year": entry.year,
        "provider": entry.provider,
        "alias": entry.alias,
        "len_chars": entry.len_chars,
        "ocrqa": entry.ocrqa,
        "tp": rebuilt.get("tp"),
        "ft": ft,
        "sents": rebuilt.get("sents"),
    }
    lingproc_path = rebuilt.get("lingproc_path")
    if lingproc_path is not None:
        out["lingproc_path"] = lingproc_path
    return out, reconstructed


def _fetch_one_file(
    bucket: str,
    key: str,
    wanted: dict[str, ManifestEntry],
) -> dict[str, tuple[dict[str, Any], bool]]:
    """Stream one rebuilt shard and pick out the manifest's ci_ids.

    Returns ``{ci_id: (output_record, ft_reconstructed)}``. The shard is
    streamed line-by-line via :func:`io.iter_jsonl_bz2` (a single
    ``Object.get()['Body']`` read piped through ``bz2.open`` in text mode),
    and the loop breaks as soon as every wanted ci_id has been picked up.

    Streaming + early-break beats multipart-parallel ``download_to_local``
    on this workload because the manifest's median 1-record-per-shard
    means we never need to read past the first matching line, while
    multipart insists on fetching the whole yearly file before any record
    can be scanned. The production pipeline embeds *every* record per
    file, where multipart wins; for the long-doc-corpus assembly we
    actively want the opposite trade-off.
    """
    found: dict[str, tuple[dict[str, Any], bool]] = {}
    remaining = set(wanted)
    for line in s3io.iter_jsonl_bz2(bucket, key):
        if not remaining:
            break
        try:
            rec = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        ci_id = rec.get("id")
        if not isinstance(ci_id, str) or ci_id not in remaining:
            continue
        out, reconstructed = _build_output_record(wanted[ci_id], rec)
        found[ci_id] = (out, reconstructed)
        remaining.discard(ci_id)
    return found


def fetch_corpus(
    manifest_path: Path,
    local_output: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> FetchStats:
    """Run the fetch and write ``local_output`` (``.jsonl.bz2``).

    Streams each unique rebuilt shard once in a thread pool, then writes the
    output in the manifest's original order so the downstream sweep sees a
    deterministic shard. Any manifest ci_id not present in its rebuilt source
    is logged as a warning and tracked in the returned stats; missing rows do
    NOT abort the run — partial coverage is more useful than no shard for an
    interactive research workflow.
    """
    entries = read_manifest(manifest_path)
    stats = FetchStats(manifest_entries=len(entries))
    if not entries:
        log.warning("manifest %s is empty; nothing to fetch", manifest_path)
        local_output.parent.mkdir(parents=True, exist_ok=True)
        with bz2.open(local_output, "wb") as fh:  # touch
            fh.write(b"")
        return stats

    groups = _group_by_source(entries)
    stats.rebuilt_files = len(groups)
    log.info(
        "planned: manifest_entries=%d rebuilt_files=%d max_workers=%d",
        stats.manifest_entries,
        stats.rebuilt_files,
        max_workers,
    )

    fetched: dict[str, tuple[dict[str, Any], bool]] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, max_workers), thread_name_prefix="corpus-fetch"
    ) as pool:
        futures = {
            pool.submit(_fetch_one_file, bucket, key, wanted): (bucket, key)
            for (bucket, key), wanted in groups.items()
        }
        with tqdm(
            total=len(futures),
            file=sys.stderr,
            disable=None,
            desc="fetch",
            unit="file",
        ) as pbar:
            for fut in as_completed(futures):
                bucket, key = futures[fut]
                try:
                    found = fut.result()
                except Exception:
                    log.exception("fetch failed for s3://%s/%s", bucket, key)
                    pbar.update(1)
                    continue
                for ci_id, payload in found.items():
                    fetched[ci_id] = payload
                pbar.update(1)

    local_output.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(local_output, "wb") as fh:
        for entry in entries:
            payload = fetched.get(entry.ci_id)
            if payload is None:
                stats.missing_in_rebuilt.append(entry.ci_id)
                log.warning(
                    "ci_id %s not found in s3://%s/%s",
                    entry.ci_id,
                    entry.rebuilt_bucket,
                    entry.rebuilt_key,
                )
                continue
            record, reconstructed = payload
            if reconstructed:
                stats.ft_reconstructed += 1
            else:
                stats.ft_from_record += 1
            fh.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
            stats.written += 1
    return stats


def _format_stats(stats: FetchStats) -> str:
    return (
        f"manifest_entries={stats.manifest_entries} "
        f"rebuilt_files={stats.rebuilt_files} "
        f"written={stats.written} "
        f"ft_from_record={stats.ft_from_record} "
        f"ft_reconstructed={stats.ft_reconstructed} "
        f"missing={len(stats.missing_in_rebuilt)}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-research-corpus-fetch",
        description=(
            "Fetch the rebuilt full-text for every entry in a chunking-eval "
            "corpus manifest and write a single .jsonl.bz2 shard, optionally "
            "uploading it to S3."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Study YAML (e.g. configs/research/study-v1.yaml). "
            "Source of truth for manifest path, output bucket, and S3 key — "
            "no per-flag overrides."
        ),
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "skip the S3 upload; the shard lands at the study's local "
            "mirror (paths.local_root + corpus.jsonl.bz2) so the user can "
            "find it deterministically."
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"thread-pool size for parallel S3 streams (default: {DEFAULT_MAX_WORKERS})",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="root log level (default: INFO)",
    )
    return p


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in dependencies
        return
    load_dotenv()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _load_env()
    cfg = load_study_config(args.config)

    manifest_path = cfg.local_path(MANIFEST_FILENAME)
    bucket = cfg.s3.bucket
    out_key = cfg.s3_key(CORPUS_FILENAME)
    out_local_mirror = cfg.local_path(CORPUS_FILENAME)

    log.info(
        "corpus-fetch start: study=%s manifest=%s output=s3://%s/%s upload=%s",
        cfg.study.name,
        manifest_path,
        bucket,
        out_key,
        "no" if args.no_upload else "yes",
    )

    with staged_output(
        bucket, out_key, out_local_mirror, upload=not args.no_upload
    ) as out_path:
        stats = fetch_corpus(
            manifest_path=manifest_path,
            local_output=out_path,
            max_workers=args.max_workers,
        )
        log.info("corpus-fetch stats: %s", _format_stats(stats))
        if stats.missing_in_rebuilt:
            log.warning(
                "%d manifest entries had no matching rebuilt record; "
                "first few: %s",
                len(stats.missing_in_rebuilt),
                stats.missing_in_rebuilt[:5],
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
