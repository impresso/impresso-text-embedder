"""CSV export for ``ValidationReport``.

Writes two flat CSVs into a directory:

* ``above_threshold.csv`` — one row per ``MismatchKind.VALUE`` mismatch
  (cosine distance > tolerance).
* ``missing.csv`` — one row per ``MISSING_IN_TARGET`` / ``MISSING_IN_PRODUCED``
  mismatch, with a ``direction`` column.

Each row's second column is a clickable Impresso article URL built from
``ci_id`` so the operator can open the article straight from the CSV.

Source-stats columns (``lg``, ``tp``, ``char_length``) are populated only
when ``--source`` was passed to the validate CLI; otherwise blank.

Dropped on purpose:

* ``tol`` — global, identical on every row, lives on the CLI invocation.
* ``reconstructable`` / ``empty`` / ``below_min_char`` — already surfaced
  as aggregates in the source-stats panel; per-row 0/1 flags duplicate
  signal that's better consumed in aggregate.

``item_id`` / ``id_key`` are added **only** when at least one row has them
populated (i.e. sentence/chunk-level reports). For text-level reports the
columns are omitted entirely so the CSV stays narrow.
"""

from __future__ import annotations

import csv
from pathlib import Path

from impresso_text_embedder.validate import (
    Mismatch,
    MismatchKind,
    SourceRecordStats,
    ValidationReport,
)

DEFAULT_URL_TEMPLATE = "https://impresso-project.ch/app/article/{ci_id}"

_BASE_ABOVE: tuple[str, ...] = ("ci_id", "url", "distance")
_BASE_MISSING: tuple[str, ...] = ("ci_id", "url", "direction")
_SOURCE_COLS: tuple[str, ...] = ("lg", "tp", "char_length")
_ITEM_COLS: tuple[str, ...] = ("item_id", "id_key")

_DIRECTION_LABELS: dict[MismatchKind, str] = {
    MismatchKind.MISSING_IN_TARGET: "missing_in_target",
    MismatchKind.MISSING_IN_PRODUCED: "missing_in_produced",
}


def _source_lookup(report: ValidationReport) -> dict[str, SourceRecordStats]:
    """Flatten ``report.source_stats`` blocks to a single ci_id → stats map.

    Direction blocks are disjoint by construction (a given ``ci_id`` lives
    in exactly one of VALUE / MISSING_IN_TARGET / MISSING_IN_PRODUCED), so
    a flat lookup is unambiguous.
    """
    if report.source_stats is None:
        return {}
    out: dict[str, SourceRecordStats] = {}
    for block in report.source_stats.blocks.values():
        out.update(block.records)
    return out


def _has_item_ids(mismatches: list[Mismatch]) -> bool:
    return any(m.item_id is not None or m.id_key is not None for m in mismatches)


def _source_columns(stats: SourceRecordStats | None) -> dict[str, str]:
    if stats is None:
        return {"lg": "", "tp": "", "char_length": ""}
    return {
        "lg": stats.lg or "",
        "tp": stats.tp or "",
        "char_length": str(stats.char_length),
    }


def _item_columns(m: Mismatch) -> dict[str, str]:
    return {
        "item_id": "" if m.item_id is None else str(m.item_id),
        "id_key": m.id_key or "",
    }


def _row_above(m: Mismatch, stats: SourceRecordStats | None, url_template: str) -> dict[str, str]:
    return {
        "ci_id": m.ci_id,
        "url": url_template.format(ci_id=m.ci_id),
        "distance": "" if m.distance is None else f"{m.distance:.6f}",
        **_source_columns(stats),
        **_item_columns(m),
    }


def _row_missing(m: Mismatch, stats: SourceRecordStats | None, url_template: str) -> dict[str, str]:
    return {
        "ci_id": m.ci_id,
        "url": url_template.format(ci_id=m.ci_id),
        "direction": _DIRECTION_LABELS[m.kind],
        **_source_columns(stats),
        **_item_columns(m),
    }


def export_report_to_csv(
    report: ValidationReport,
    output_dir: Path | str,
    *,
    url_template: str = DEFAULT_URL_TEMPLATE,
) -> tuple[Path, Path]:
    """Write ``above_threshold.csv`` and ``missing.csv`` into ``output_dir``.

    Both files are always written — even if a bucket is empty, the file
    lands with just a header row, which keeps downstream tooling that
    globs the directory predictable.

    ``item_id`` / ``id_key`` columns are included only when at least one
    row in that bucket carries them (sentence/chunk-level reports). Text-
    level reports get the narrower 6-column layout.

    Returns the two paths actually written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    above_path = out / "above_threshold.csv"
    missing_path = out / "missing.csv"

    lookup = _source_lookup(report)

    above_mms = [m for m in report.mismatches if m.kind == MismatchKind.VALUE]
    missing_mms = [m for m in report.mismatches if m.kind in _DIRECTION_LABELS]

    above_fields = list(_BASE_ABOVE) + list(_SOURCE_COLS)
    if _has_item_ids(above_mms):
        above_fields += list(_ITEM_COLS)
    missing_fields = list(_BASE_MISSING) + list(_SOURCE_COLS)
    if _has_item_ids(missing_mms):
        missing_fields += list(_ITEM_COLS)

    with above_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=above_fields, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for m in above_mms:
            writer.writerow(_row_above(m, lookup.get(m.ci_id), url_template))

    with missing_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=missing_fields, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for m in missing_mms:
            writer.writerow(_row_missing(m, lookup.get(m.ci_id), url_template))

    return above_path, missing_path
