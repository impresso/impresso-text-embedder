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
