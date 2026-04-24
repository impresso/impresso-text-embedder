"""Validation helpers for embedding output files.

Two modes:
  * structural — lines parse, fields present, embeddings same length, no NaN/Inf;
  * comparison against a target file — cosine distance per matching record/item.

See ``.progress/validation-metric/notes.md`` for the metric choice.
"""

from __future__ import annotations

import bz2
import json
import logging
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from impresso_text_embedder import io as s3io

log = logging.getLogger(__name__)

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
DEFAULT_TOL = 1e-4


@dataclass
class ValidationReport:
    records_checked: int = 0
    items_checked: int = 0
    max_distance: float = 0.0
    mismatches: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

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
            yield json.loads(line)
        except json.JSONDecodeError as exc:
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
            report.mismatches.append(f"missing in produced: ci_id={rid!r}")
            continue
        if rid not in e_idx:
            report.mismatches.append(f"missing in target: ci_id={rid!r}")
            continue
        d = _cosine_distance(p_idx[rid]["embedding"], e_idx[rid]["embedding"])
        if d > tol:
            report.mismatches.append(f"ci_id={rid!r}: cosine distance {d:.3e} > tol {tol:.0e}")
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
            report.mismatches.append(f"missing in produced: ci_id={ci_id!r} {id_key}={item_id}")
            continue
        if key not in e_idx:
            report.mismatches.append(f"missing in target: ci_id={ci_id!r} {id_key}={item_id}")
            continue
        d = _cosine_distance(p_idx[key]["embedding"], e_idx[key]["embedding"])
        if d > tol:
            report.mismatches.append(
                f"ci_id={ci_id!r} {id_key}={item_id}: cosine distance {d:.3e} > tol {tol:.0e}"
            )
        report.max_distance = max(report.max_distance, d)
        report.items_checked += 1
        seen_records.add(ci_id)
    report.records_checked = len(seen_records)
