"""Validation helpers for embedding output files.

Two modes:
  * structural — lines parse, fields present, embeddings same length, no NaN/Inf;
  * comparison against a target file — cosine distance per matching record/item.

See ``.progress/validation-metric/notes.md`` for the metric choice.
"""

from __future__ import annotations

import bz2
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import orjson

from impresso_text_embedder import io as s3io
from impresso_text_embedder.text import rebuild_ft_from_offsets

log = logging.getLogger(__name__)

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
DEFAULT_TOL = 1e-4


class MismatchKind(str, Enum):
    VALUE = "value"
    MISSING_IN_TARGET = "missing_in_target"
    MISSING_IN_PRODUCED = "missing_in_produced"


@dataclass
class Mismatch:
    kind: MismatchKind
    ci_id: str
    item_id: int | str | None = None
    id_key: str | None = None
    distance: float | None = None
    tol: float | None = None

    def _qualifier(self) -> str:
        if self.id_key is not None and self.item_id is not None:
            return f"ci_id={self.ci_id!r} {self.id_key}={self.item_id}"
        return f"ci_id={self.ci_id!r}"

    def __str__(self) -> str:
        q = self._qualifier()
        if self.kind == MismatchKind.MISSING_IN_PRODUCED:
            return f"missing in produced: {q}"
        if self.kind == MismatchKind.MISSING_IN_TARGET:
            return f"missing in target: {q}"
        tol = self.tol if self.tol is not None else DEFAULT_TOL
        d = self.distance if self.distance is not None else float("nan")
        return f"{q}: cosine distance {d:.3e} > tol {tol:.0e}"


DEFAULT_SOURCE_MIN_CHAR_LENGTH = 400
DEFAULT_SAMPLE_EXCERPT_COUNT = 3
DEFAULT_EXCERPT_CHARS = 80


@dataclass
class Sample:
    """One source record displayed alongside the stats block.

    ``distance`` is populated for the above-tolerance (``MismatchKind.VALUE``)
    direction with the record-level max cosine distance across its items.
    For the two missing directions there is no distance by construction —
    the record isn't on both sides — so the field stays ``None``.
    """

    ci_id: str
    excerpt: str
    distance: float | None = None


@dataclass
class SourceStatsBlock:
    """Source-backed diagnostics for one mismatch direction.

    Fields tally what could be learned by cross-referencing the ids
    in a single ``MismatchKind`` bucket against the original input
    ``.jsonl.bz2``.
    """

    direction: MismatchKind
    total: int = 0
    found_in_source: int = 0
    not_in_source_ids: list[str] = field(default_factory=list)
    reconstructable: int = 0
    empty: int = 0
    below_min_char: int = 0
    char_lengths: list[int] = field(default_factory=list)
    lg_counts: Counter[str] = field(default_factory=Counter)
    tp_counts: Counter[str] = field(default_factory=Counter)
    samples: list[Sample] = field(default_factory=list)


@dataclass
class SourceStatsAnalysis:
    """Per-direction source analysis attached to a ``ValidationReport``."""

    min_char_length: int = DEFAULT_SOURCE_MIN_CHAR_LENGTH
    blocks: dict[MismatchKind, SourceStatsBlock] = field(default_factory=dict)


@dataclass
class ValidationReport:
    level: str | None = None
    records_checked: int = 0
    items_checked: int = 0
    max_distance: float = 0.0
    distances: list[float] = field(default_factory=list)
    mismatches: list[Mismatch] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source_stats: SourceStatsAnalysis | None = None

    @property
    def passed(self) -> bool:
        return not self.errors and not self.mismatches


# ---------------------------------------------------------------------------
# Line iteration
# ---------------------------------------------------------------------------


def iter_lines_from_path(path: str | Path) -> Iterator[str]:
    """Yield lines from a `.jsonl.bz2` file at an S3 URI or local path."""
    p = str(path)
    if p.startswith("s3://"):
        bucket, key = s3io.parse_s3_uri(p)
        yield from s3io.iter_jsonl_bz2(bucket, key)
        return

    with bz2.open(p, "rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line:
                yield line


# ---------------------------------------------------------------------------
# Parsing / level detection
# ---------------------------------------------------------------------------


def detect_level(record: dict) -> str:
    """Return ``text``, ``sentence``, or ``chunk`` based on record shape."""
    if "sents" in record and isinstance(record["sents"], list):
        return "sentence"
    if "chunks" in record and isinstance(record["chunks"], list):
        return "chunk"
    if "embedding" in record and isinstance(record["embedding"], list):
        return "text"
    raise ValueError("record does not match any known level shape")


def parse_records(lines: Iterable[str]) -> Iterator[dict]:
    for i, line in enumerate(lines, start=1):
        try:
            yield orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"line {i}: malformed JSON ({exc})") from exc


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def _finite(vec: list[float]) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in vec)


def validate_structural(path: str | Path) -> ValidationReport:
    report = ValidationReport()
    expected_dim: int | None = None
    detected_level: str | None = None
    try:
        records = list(parse_records(iter_lines_from_path(path)))
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    for i, rec in enumerate(records, start=1):
        try:
            level = detect_level(rec)
        except ValueError as exc:
            report.errors.append(f"record {i}: {exc}")
            continue
        if detected_level is None:
            detected_level = level
        elif level != detected_level:
            report.errors.append(
                f"record {i}: level {level} differs from file-wide {detected_level}"
            )
            continue

        if "ts" in rec and not TS_RE.match(rec["ts"]):
            report.errors.append(f"record {i}: bad ts {rec['ts']!r}")

        if level == "text":
            expected_dim = _check_text_record(rec, i, report, expected_dim=expected_dim)
            report.records_checked += 1
            report.items_checked += 1
        elif level == "sentence":
            expected_dim = _check_item_list_record(
                rec, key="sents", id_key="sent_id", index=i, report=report, expected_dim=expected_dim
            )
            report.records_checked += 1
            report.items_checked += len(rec.get("sents") or [])
        elif level == "chunk":
            expected_dim = _check_item_list_record(
                rec, key="chunks", id_key="chunk_id", index=i, report=report, expected_dim=expected_dim
            )
            report.records_checked += 1
            report.items_checked += len(rec.get("chunks") or [])

    report.level = detected_level
    return report


def _check_text_record(rec, index, report, *, expected_dim):
    for req in ("ci_id", "model_id", "embedding", "size"):
        if req not in rec:
            report.errors.append(f"record {index}: text record missing {req!r}")
            return expected_dim
    emb = rec["embedding"]
    if not isinstance(emb, list) or not emb:
        report.errors.append(f"record {index}: empty or non-list embedding")
        return expected_dim
    if not _finite(emb):
        report.errors.append(f"record {index}: embedding has non-finite values")
    size = rec["size"]
    if not isinstance(size, int) or size != len(emb):
        report.errors.append(
            f"record {index}: size {size!r} != len(embedding) {len(emb)}"
        )
    if expected_dim is None:
        expected_dim = len(emb)
    elif len(emb) != expected_dim:
        report.errors.append(
            f"record {index}: embedding dim {len(emb)} != file dim {expected_dim}"
        )
    return expected_dim


def _check_item_list_record(rec, *, key, id_key, index, report, expected_dim):
    for req in ("ci_id", key):
        if req not in rec:
            report.errors.append(f"record {index}: missing {req!r}")
            return expected_dim
    items = rec.get(key) or []
    if not isinstance(items, list):
        report.errors.append(f"record {index}: {key!r} is not a list")
        return expected_dim
    for j, item in enumerate(items):
        if not isinstance(item, dict):
            report.errors.append(f"record {index} item {j}: not a dict")
            continue
        if id_key not in item or "embedding" not in item:
            report.errors.append(
                f"record {index} item {j}: missing {id_key!r} or 'embedding'"
            )
            continue
        emb = item["embedding"]
        if not isinstance(emb, list) or not emb:
            report.errors.append(
                f"record {index} item {j}: empty or non-list embedding"
            )
            continue
        if not _finite(emb):
            report.errors.append(
                f"record {index} item {j}: embedding has non-finite values"
            )
        if expected_dim is None:
            expected_dim = len(emb)
        elif len(emb) != expected_dim:
            report.errors.append(
                f"record {index} item {j}: dim {len(emb)} != file dim {expected_dim}"
            )
    return expected_dim


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    cos = dot / (na * nb)
    # Clamp for numerical robustness before subtracting.
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return 1.0 - cos


def _index_text(records: Iterable[dict]) -> dict[str, dict]:
    return {r["ci_id"]: r for r in records if "ci_id" in r}


def _index_items(records: Iterable[dict], list_key: str, id_key: str) -> dict[tuple[str, int], dict]:
    idx = {}
    for r in records:
        ci_id = r.get("ci_id")
        if ci_id is None:
            continue
        for item in r.get(list_key) or []:
            item_id = item.get(id_key)
            if item_id is None:
                continue
            idx[(ci_id, item_id)] = item
    return idx


def validate_against_target(
    path: str | Path,
    target: str | Path,
    tol: float = DEFAULT_TOL,
) -> ValidationReport:
    report = ValidationReport()

    try:
        produced = list(parse_records(iter_lines_from_path(path)))
        expected = list(parse_records(iter_lines_from_path(target)))
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    if not produced and not expected:
        return report
    if not produced or not expected:
        report.errors.append("one side is empty while the other is not")
        return report

    try:
        level = detect_level(produced[0])
        level_t = detect_level(expected[0])
    except ValueError as exc:
        report.errors.append(str(exc))
        return report
    if level != level_t:
        report.errors.append(f"level mismatch: produced={level} target={level_t}")
        return report

    report.level = level
    if level == "text":
        _compare_text(produced, expected, tol, report)
    elif level == "sentence":
        _compare_items(produced, expected, tol, report, list_key="sents", id_key="sent_id")
    elif level == "chunk":
        _compare_items(produced, expected, tol, report, list_key="chunks", id_key="chunk_id")
    return report


def _compare_text(produced, expected, tol, report):
    p_idx = _index_text(produced)
    e_idx = _index_text(expected)
    all_ids = set(p_idx) | set(e_idx)
    for rid in sorted(all_ids):
        if rid not in p_idx:
            report.mismatches.append(Mismatch(MismatchKind.MISSING_IN_PRODUCED, ci_id=rid))
            continue
        if rid not in e_idx:
            report.mismatches.append(Mismatch(MismatchKind.MISSING_IN_TARGET, ci_id=rid))
            continue
        d = _cosine_distance(p_idx[rid]["embedding"], e_idx[rid]["embedding"])
        report.distances.append(d)
        if d > tol:
            report.mismatches.append(
                Mismatch(MismatchKind.VALUE, ci_id=rid, distance=d, tol=tol)
            )
        report.max_distance = max(report.max_distance, d)
        report.records_checked += 1
        report.items_checked += 1


def _compare_items(produced, expected, tol, report, *, list_key, id_key):
    p_idx = _index_items(produced, list_key=list_key, id_key=id_key)
    e_idx = _index_items(expected, list_key=list_key, id_key=id_key)
    all_keys = set(p_idx) | set(e_idx)
    seen_records: set[str] = set()
    for key in sorted(all_keys):
        ci_id, item_id = key
        if key not in p_idx:
            report.mismatches.append(
                Mismatch(
                    MismatchKind.MISSING_IN_PRODUCED,
                    ci_id=ci_id,
                    item_id=item_id,
                    id_key=id_key,
                )
            )
            continue
        if key not in e_idx:
            report.mismatches.append(
                Mismatch(
                    MismatchKind.MISSING_IN_TARGET,
                    ci_id=ci_id,
                    item_id=item_id,
                    id_key=id_key,
                )
            )
            continue
        d = _cosine_distance(p_idx[key]["embedding"], e_idx[key]["embedding"])
        report.distances.append(d)
        if d > tol:
            report.mismatches.append(
                Mismatch(
                    MismatchKind.VALUE,
                    ci_id=ci_id,
                    item_id=item_id,
                    id_key=id_key,
                    distance=d,
                    tol=tol,
                )
            )
        report.max_distance = max(report.max_distance, d)
        report.items_checked += 1
        seen_records.add(ci_id)
    report.records_checked = len(seen_records)


# ---------------------------------------------------------------------------
# Source-backed diagnostics
# ---------------------------------------------------------------------------


_SOURCE_DIRECTIONS: tuple[MismatchKind, ...] = (
    MismatchKind.VALUE,
    MismatchKind.MISSING_IN_TARGET,
    MismatchKind.MISSING_IN_PRODUCED,
)


def _target_ids_by_direction(
    mismatches: Iterable[Mismatch],
) -> dict[MismatchKind, set[str]]:
    """Extract the set of ``ci_id``s per mismatch-direction bucket.

    A single record can yield several item-level mismatches (sentence or
    chunk level), but the source-stats layer operates at record
    granularity so we collapse duplicates.
    """
    out: dict[MismatchKind, set[str]] = {k: set() for k in _SOURCE_DIRECTIONS}
    for m in mismatches:
        if m.kind in out:
            out[m.kind].add(m.ci_id)
    return out


def _value_distance_per_record(
    mismatches: Iterable[Mismatch],
) -> dict[str, float]:
    """Record-level max cosine distance across item-level ``VALUE`` mismatches.

    Used to (a) rank the above-tolerance samples by worst drift and (b)
    annotate each sample with its distance in the rendered panel.
    """
    out: dict[str, float] = {}
    for m in mismatches:
        if m.kind != MismatchKind.VALUE or m.distance is None:
            continue
        prev = out.get(m.ci_id)
        if prev is None or m.distance > prev:
            out[m.ci_id] = m.distance
    return out


def _reconstruct_text(record: dict) -> str:
    ft = record.get("ft")
    if isinstance(ft, str) and ft:
        return ft
    sents = record.get("sents")
    if isinstance(sents, list) and sents:
        try:
            return rebuild_ft_from_offsets(sents)
        except (KeyError, TypeError):
            return ""
    return ""


def _record_for_ci_id(record: dict) -> str | None:
    rid = record.get("id")
    return rid if isinstance(rid, str) else None


def collect_source_stats(
    source_path: str | Path,
    report: ValidationReport,
    min_char_length: int = DEFAULT_SOURCE_MIN_CHAR_LENGTH,
    sample_count: int = DEFAULT_SAMPLE_EXCERPT_COUNT,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> SourceStatsAnalysis:
    """Stream ``source_path`` and tally source-backed stats for every
    ``ci_id`` that shows up as ``VALUE`` (above tolerance),
    ``MISSING_IN_TARGET``, or ``MISSING_IN_PRODUCED`` in
    ``report.mismatches``.

    The result is also attached to ``report.source_stats`` so callers
    (CLI renderer, tests) can inspect it off the report alone.

    Sample ranking:
      * ``VALUE``: top ``sample_count`` by record-level max cosine
        distance, descending (worst drift first). Each ``Sample`` carries
        its ``distance``.
      * ``MISSING_IN_*``: first ``sample_count`` seen while scanning the
        source. ``Sample.distance`` is ``None`` by construction — the
        record isn't on both sides.
    """
    buckets = _target_ids_by_direction(report.mismatches)
    value_distances = _value_distance_per_record(report.mismatches)
    analysis = SourceStatsAnalysis(min_char_length=min_char_length)
    for direction in _SOURCE_DIRECTIONS:
        analysis.blocks[direction] = SourceStatsBlock(
            direction=direction,
            total=len(buckets[direction]),
        )

    # Nothing to look up — early return with empty blocks.
    wanted = set().union(*buckets.values())
    if not wanted:
        report.source_stats = analysis
        return analysis

    # Reverse-map each ci_id → direction it belongs to. The three
    # direction sets are disjoint by construction (VALUE means both
    # sides have the record but differ; MISSING_* means exactly one
    # side has it).
    lookup: dict[str, MismatchKind] = {}
    for direction, ids in buckets.items():
        for cid in ids:
            lookup[cid] = direction

    # VALUE samples are picked at finalise time (top by distance) so we
    # buffer every candidate rather than cutting off at ``sample_count``
    # during the scan.
    value_candidates: list[Sample] = []

    remaining = set(wanted)
    for raw in parse_records(iter_lines_from_path(source_path)):
        rid = _record_for_ci_id(raw)
        if rid is None or rid not in lookup:
            continue
        direction = lookup[rid]
        block = analysis.blocks[direction]

        text = _reconstruct_text(raw)
        length = len(text)
        has_sents = isinstance(raw.get("sents"), list) and bool(raw.get("sents"))
        has_ft = isinstance(raw.get("ft"), str) and bool(raw["ft"])

        block.found_in_source += 1
        if has_sents or has_ft:
            block.reconstructable += 1
        else:
            block.empty += 1
        if length < min_char_length:
            block.below_min_char += 1
        block.char_lengths.append(length)

        lg = raw.get("lg") if isinstance(raw.get("lg"), str) else None
        block.lg_counts[lg or "(missing)"] += 1
        tp = raw.get("tp") if isinstance(raw.get("tp"), str) else None
        block.tp_counts[tp or "(missing)"] += 1

        excerpt = text[:excerpt_chars]
        if direction == MismatchKind.VALUE:
            value_candidates.append(
                Sample(ci_id=rid, excerpt=excerpt, distance=value_distances.get(rid))
            )
        elif len(block.samples) < sample_count:
            block.samples.append(Sample(ci_id=rid, excerpt=excerpt))

        remaining.discard(rid)
        if not remaining:
            break

    # Top-N worst drifts for the VALUE panel. ``None`` distances (should
    # not happen in practice) sort to the end.
    if value_candidates:
        value_candidates.sort(
            key=lambda s: (s.distance if s.distance is not None else float("-inf")),
            reverse=True,
        )
        analysis.blocks[MismatchKind.VALUE].samples = value_candidates[:sample_count]

    # Whatever didn't turn up in the source goes in the drift bucket.
    for rid in sorted(remaining):
        direction = lookup[rid]
        analysis.blocks[direction].not_in_source_ids.append(rid)

    report.source_stats = analysis
    return analysis
