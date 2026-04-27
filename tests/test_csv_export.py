from __future__ import annotations

import csv
from pathlib import Path

from impresso_text_embedder.csv_export import (
    DEFAULT_URL_TEMPLATE,
    export_report_to_csv,
)
from impresso_text_embedder.validate import (
    Mismatch,
    MismatchKind,
    SourceRecordStats,
    SourceStatsAnalysis,
    SourceStatsBlock,
    ValidationReport,
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _build_text_report_with_source() -> ValidationReport:
    """Text-level report — no item_id/id_key on any mismatch."""
    report = ValidationReport(level="text", records_checked=4, items_checked=4)
    report.mismatches = [
        Mismatch(MismatchKind.VALUE, ci_id="ci-drift-1", distance=0.0123, tol=1e-4),
        Mismatch(MismatchKind.VALUE, ci_id="ci-drift-2", distance=0.5, tol=1e-4),
        Mismatch(MismatchKind.MISSING_IN_TARGET, ci_id="ci-miss-target"),
        Mismatch(MismatchKind.MISSING_IN_PRODUCED, ci_id="ci-miss-produced"),
    ]
    blocks: dict[MismatchKind, SourceStatsBlock] = {}
    blocks[MismatchKind.VALUE] = SourceStatsBlock(direction=MismatchKind.VALUE, total=2)
    blocks[MismatchKind.VALUE].records["ci-drift-1"] = SourceRecordStats(
        char_length=520, lg="fr", tp="ar", reconstructable=True, empty=False, below_min_char=False
    )
    blocks[MismatchKind.VALUE].records["ci-drift-2"] = SourceRecordStats(
        char_length=120, lg="de", tp=None, reconstructable=True, empty=False, below_min_char=True
    )
    blocks[MismatchKind.MISSING_IN_TARGET] = SourceStatsBlock(
        direction=MismatchKind.MISSING_IN_TARGET, total=1
    )
    blocks[MismatchKind.MISSING_IN_TARGET].records["ci-miss-target"] = SourceRecordStats(
        char_length=900, lg="en", tp="ar", reconstructable=True, empty=False, below_min_char=False
    )
    blocks[MismatchKind.MISSING_IN_PRODUCED] = SourceStatsBlock(
        direction=MismatchKind.MISSING_IN_PRODUCED, total=1
    )
    # ci-miss-produced left out of records → "not in source" case (blank source columns).
    report.source_stats = SourceStatsAnalysis(min_char_length=400, blocks=blocks)
    return report


def _build_item_level_report() -> ValidationReport:
    """Sentence-level report — item_id/id_key populated, no source stats."""
    report = ValidationReport(level="sentence", records_checked=2, items_checked=3)
    report.mismatches = [
        Mismatch(
            MismatchKind.VALUE,
            ci_id="ci-1",
            item_id=7,
            id_key="sent_id",
            distance=0.002,
            tol=1e-4,
        ),
        Mismatch(MismatchKind.MISSING_IN_TARGET, ci_id="ci-2", item_id=3, id_key="sent_id"),
    ]
    return report


def _build_text_report_no_source() -> ValidationReport:
    report = ValidationReport(level="text", records_checked=1, items_checked=1)
    report.mismatches = [
        Mismatch(MismatchKind.VALUE, ci_id="ci-x", distance=0.002, tol=1e-4),
    ]
    return report


def test_export_writes_both_files(tmp_path):
    above, missing = export_report_to_csv(_build_text_report_with_source(), tmp_path)
    assert above == tmp_path / "above_threshold.csv"
    assert missing == tmp_path / "missing.csv"
    assert above.exists() and missing.exists()


def test_export_creates_output_dir(tmp_path):
    nested = tmp_path / "deep" / "nested" / "out"
    above, missing = export_report_to_csv(_build_text_report_with_source(), nested)
    assert above.exists() and missing.exists()


def test_text_level_above_threshold_schema(tmp_path):
    """Text-level reports get the narrow 6-column layout — no item_id/id_key."""
    export_report_to_csv(_build_text_report_with_source(), tmp_path)
    fieldnames, rows = _read_csv(tmp_path / "above_threshold.csv")

    assert fieldnames == ["ci_id", "url", "distance", "lg", "tp", "char_length"]
    # Dropped columns must not reappear.
    for dead in ("tol", "reconstructable", "empty", "below_min_char", "item_id", "id_key"):
        assert dead not in fieldnames

    by_id = {r["ci_id"]: r for r in rows}
    assert set(by_id) == {"ci-drift-1", "ci-drift-2"}
    assert by_id["ci-drift-1"]["url"] == DEFAULT_URL_TEMPLATE.format(ci_id="ci-drift-1")
    assert float(by_id["ci-drift-1"]["distance"]) == 0.0123
    assert by_id["ci-drift-1"]["lg"] == "fr"
    assert by_id["ci-drift-1"]["tp"] == "ar"
    assert by_id["ci-drift-1"]["char_length"] == "520"
    # tp absent in source → blank cell, not "(missing)".
    assert by_id["ci-drift-2"]["tp"] == ""


def test_text_level_missing_schema(tmp_path):
    export_report_to_csv(_build_text_report_with_source(), tmp_path)
    fieldnames, rows = _read_csv(tmp_path / "missing.csv")

    assert fieldnames == ["ci_id", "url", "direction", "lg", "tp", "char_length"]
    for dead in ("tol", "reconstructable", "empty", "below_min_char", "item_id", "id_key", "distance"):
        assert dead not in fieldnames

    by_id = {r["ci_id"]: r for r in rows}
    assert by_id["ci-miss-target"]["direction"] == "missing_in_target"
    assert by_id["ci-miss-produced"]["direction"] == "missing_in_produced"
    # Source-stats present.
    assert by_id["ci-miss-target"]["lg"] == "en"
    assert by_id["ci-miss-target"]["char_length"] == "900"
    # Not in source → blank source columns.
    assert by_id["ci-miss-produced"]["lg"] == ""
    assert by_id["ci-miss-produced"]["char_length"] == ""


def test_item_level_adds_item_columns(tmp_path):
    """Sentence/chunk-level reports get item_id/id_key appended."""
    export_report_to_csv(_build_item_level_report(), tmp_path)

    above_fields, above_rows = _read_csv(tmp_path / "above_threshold.csv")
    miss_fields, miss_rows = _read_csv(tmp_path / "missing.csv")

    assert above_fields == [
        "ci_id", "url", "distance", "lg", "tp", "char_length", "item_id", "id_key",
    ]
    assert miss_fields == [
        "ci_id", "url", "direction", "lg", "tp", "char_length", "item_id", "id_key",
    ]

    assert above_rows[0]["item_id"] == "7"
    assert above_rows[0]["id_key"] == "sent_id"
    assert miss_rows[0]["item_id"] == "3"
    assert miss_rows[0]["id_key"] == "sent_id"


def test_export_without_source_blanks_source_columns(tmp_path):
    export_report_to_csv(_build_text_report_no_source(), tmp_path)
    fieldnames, rows = _read_csv(tmp_path / "above_threshold.csv")
    # Schema is identical with/without source — only the cells differ.
    assert fieldnames == ["ci_id", "url", "distance", "lg", "tp", "char_length"]
    row = rows[0]
    assert row["ci_id"] == "ci-x"
    assert row["url"] == DEFAULT_URL_TEMPLATE.format(ci_id="ci-x")
    assert float(row["distance"]) == 0.002
    assert row["lg"] == ""
    assert row["tp"] == ""
    assert row["char_length"] == ""


def test_export_with_empty_report_writes_headers_only(tmp_path):
    report = ValidationReport(level="text")
    above, missing = export_report_to_csv(report, tmp_path)

    above_fields, above_rows = _read_csv(above)
    miss_fields, miss_rows = _read_csv(missing)
    assert above_rows == []
    assert miss_rows == []
    assert above_fields == ["ci_id", "url", "distance", "lg", "tp", "char_length"]
    assert miss_fields == ["ci_id", "url", "direction", "lg", "tp", "char_length"]


def test_custom_url_template(tmp_path):
    template = "https://example.test/article/{ci_id}?utm=val"
    export_report_to_csv(
        _build_text_report_with_source(), tmp_path, url_template=template
    )
    _, rows = _read_csv(tmp_path / "above_threshold.csv")
    assert rows[0]["url"] == template.format(ci_id=rows[0]["ci_id"])
