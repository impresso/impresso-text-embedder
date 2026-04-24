from __future__ import annotations

import bz2
import json
from pathlib import Path

import pytest

from impresso_text_embedder.cli import validate as validate_cli


def _write(path: Path, records: list[dict]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r))
            fh.write("\n")


def _text(ci_id: str, emb: list[float]) -> dict:
    return {
        "ci_id": ci_id,
        "model_id": "m@default",
        "embedding": emb,
        "size": len(emb),
        "ts": "2024-01-02T03:04:05Z",
    }


def test_parser_defaults():
    args = validate_cli.build_parser().parse_args(["file.jsonl.bz2"])
    assert args.path == "file.jsonl.bz2"
    assert args.target is None
    assert args.tol == pytest.approx(1e-4)


def test_main_structural_ok(tmp_path, capsys):
    p = tmp_path / "e.jsonl.bz2"
    _write(p, [_text("a", [1.0, 0.0])])
    rc = validate_cli.main([str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "records_checked=1" in out
    assert "OK" in out


def test_main_structural_fails_on_nan(tmp_path):
    p = tmp_path / "e.jsonl.bz2"
    _write(p, [_text("a", [float("nan"), 0.0])])
    rc = validate_cli.main([str(p)])
    assert rc == 1


def test_main_self_comparison_ok(tmp_path, capsys):
    p = tmp_path / "e.jsonl.bz2"
    _write(p, [_text("a", [1.0, 0.0])])
    rc = validate_cli.main([str(p), "--target", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_main_comparison_beyond_tol_fails(tmp_path):
    a = tmp_path / "a.jsonl.bz2"
    b = tmp_path / "b.jsonl.bz2"
    _write(a, [_text("x", [1.0, 0.0])])
    _write(b, [_text("x", [0.0, 1.0])])
    rc = validate_cli.main([str(a), "--target", str(b), "--tol", "0.5"])
    assert rc == 1


def test_main_comparison_output_includes_stats_and_histogram(tmp_path, capsys):
    a = tmp_path / "a.jsonl.bz2"
    b = tmp_path / "b.jsonl.bz2"
    # Several pairs, a mix of drifts so distances + histogram are populated.
    _write(
        a,
        [
            _text("p1", [1.0, 0.0]),
            _text("p2", [1.0, 0.0]),
            _text("p3", [1.0, 0.0]),
            _text("p4", [1.0, 0.0]),
        ],
    )
    _write(
        b,
        [
            _text("p1", [1.0, 0.0]),        # 0
            _text("p2", [0.9999, 0.0141]),  # ~1e-4 drift
            _text("p3", [0.99, 0.141]),     # ~1e-2 drift
            _text("p4", [0.0, 1.0]),        # 1.0 drift
        ],
    )
    rc = validate_cli.main([str(a), "--target", str(b), "--tol", "1e-4"])
    assert rc == 1
    out = capsys.readouterr().out
    # Statistical sections.
    assert "p50=" in out
    assert "log10(distance)" in out
    assert "worst drifts" in out
    assert "mismatches" in out
    # Legacy markers still present.
    assert "records_checked=4" in out
    assert "MISMATCH:" in out
    assert "FAIL" in out


def test_main_comparison_missing_records_rendered(tmp_path, capsys):
    a = tmp_path / "a.jsonl.bz2"
    b = tmp_path / "b.jsonl.bz2"
    _write(a, [_text("x", [1.0, 0.0]), _text("y", [1.0, 0.0])])
    _write(b, [_text("x", [1.0, 0.0])])
    rc = validate_cli.main([str(a), "--target", str(b)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "missing in target (1)" in out
    assert "FAIL" in out


def test_parser_accepts_new_flags():
    args = validate_cli.build_parser().parse_args(
        ["file.jsonl.bz2", "--top", "5", "--show-all-missing"]
    )
    assert args.top == 5
    assert args.show_all_missing is True


def _source(path: Path, records: list[dict]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r))
            fh.write("\n")


def test_parser_accepts_source_flags():
    args = validate_cli.build_parser().parse_args(
        [
            "file.jsonl.bz2",
            "--source",
            "s.jsonl.bz2",
            "--source-min-char-length",
            "200",
        ]
    )
    assert args.source == "s.jsonl.bz2"
    assert args.source_min_char_length == 200


def test_parser_source_default_is_none():
    args = validate_cli.build_parser().parse_args(["file.jsonl.bz2"])
    assert args.source is None
    assert args.source_min_char_length == 400  # DEFAULT_SOURCE_MIN_CHAR_LENGTH


def test_source_renders_panel_when_target_given(tmp_path, capsys):
    a = tmp_path / "a.jsonl.bz2"
    b = tmp_path / "b.jsonl.bz2"
    source = tmp_path / "s.jsonl.bz2"
    _write(a, [_text("x", [1.0, 0.0]), _text("y", [1.0, 0.0])])
    _write(b, [_text("x", [1.0, 0.0])])
    # y is missing in target; source carries its full text.
    _source(
        source,
        [
            {"id": "y", "tp": "ar", "lg": "fr", "ft": "contenu de y " * 40},
        ],
    )

    rc = validate_cli.main(
        [
            str(a),
            "--target",
            str(b),
            "--source",
            str(source),
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    # Source-stats panel for the missing-in-target direction must appear.
    assert "source analysis" in out
    assert "missing in target" in out
    # Counts table in the panel includes the reconstructable row.
    assert "reconstructable" in out
    # Legacy markers still present.
    assert "records_checked=" in out
    assert "FAIL" in out


def test_source_without_target_warns(tmp_path, capsys):
    p = tmp_path / "e.jsonl.bz2"
    _write(p, [_text("a", [1.0, 0.0])])
    s = tmp_path / "s.jsonl.bz2"
    _source(s, [{"id": "a", "ft": "hello"}])

    rc = validate_cli.main([str(p), "--source", str(s)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "--source" in captured.err
    # Structural output still emitted, no source-stats section.
    assert "source analysis" not in captured.out


def test_above_tol_panel_renders(tmp_path, capsys):
    a = tmp_path / "a.jsonl.bz2"
    b = tmp_path / "b.jsonl.bz2"
    source = tmp_path / "s.jsonl.bz2"
    # One above-tol pair (orthogonal → distance=1), plus a passing pair.
    _write(a, [_text("d1", [1.0, 0.0]), _text("d2", [1.0, 0.0])])
    _write(b, [_text("d1", [1.0, 0.0]), _text("d2", [0.0, 1.0])])
    with bz2.open(source, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "d2", "tp": "ar", "lg": "fr", "ft": "contenu " * 40}))
        fh.write("\n")

    rc = validate_cli.main([str(a), "--target", str(b), "--source", str(source), "--tol", "1e-4"])
    assert rc == 1
    out = capsys.readouterr().out
    # New VALUE panel title + per-sample distance annotation.
    assert "above tolerance" in out
    assert "d=" in out
    # Legacy substring markers still present.
    assert "records_checked=" in out
    assert "FAIL" in out


def test_source_empty_source_still_runs(tmp_path, capsys):
    # y is missing in target; source is empty → not_in_source drift signal.
    a = tmp_path / "a.jsonl.bz2"
    b = tmp_path / "b.jsonl.bz2"
    source = tmp_path / "s.jsonl.bz2"
    _write(a, [_text("x", [1.0, 0.0]), _text("y", [1.0, 0.0])])
    _write(b, [_text("x", [1.0, 0.0])])
    _source(source, [])

    rc = validate_cli.main(
        [
            str(a),
            "--target",
            str(b),
            "--source",
            str(source),
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "source analysis" in out
    assert "not in source" in out
