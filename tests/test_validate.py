from __future__ import annotations

import bz2
import json
from pathlib import Path

import pytest

from impresso_text_embedder import validate as v


def _write_jsonl_bz2(path: Path, records: list[dict]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r))
            fh.write("\n")


def _text_record(id_: str, emb: list[float]) -> dict:
    return {
        "id": id_,
        "ts": "2024-01-02T03:04:05Z",
        "embedder": "m@default",
        "len": 100,
        "embedding": emb,
    }


def _sentence_record(ci_id: str, sents: list[tuple[int, list[float]]]) -> dict:
    return {
        "ts": "2024-01-02T03:04:05Z",
        "ci_id": ci_id,
        "sents": [{"sent_id": sid, "embedding": emb, "size": len(emb)} for sid, emb in sents],
    }


def test_cosine_distance_zero_for_equal_vectors():
    assert v._cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-12)


def test_cosine_distance_two_for_antiparallel():
    assert v._cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0, abs=1e-12)


def test_cosine_distance_invariant_to_scale():
    d1 = v._cosine_distance([1.0, 0.0], [1.0, 0.0])
    d2 = v._cosine_distance([1.0, 0.0], [1000.0, 0.0])
    assert d1 == pytest.approx(d2, abs=1e-12)


def test_cosine_distance_dim_mismatch_raises():
    with pytest.raises(ValueError):
        v._cosine_distance([1.0], [1.0, 2.0])


def test_detect_level_text():
    assert v.detect_level(_text_record("x", [1.0, 0.0])) == "text"


def test_detect_level_sentence():
    assert v.detect_level(_sentence_record("x", [(0, [1.0])])) == "sentence"


def test_detect_level_unknown():
    with pytest.raises(ValueError):
        v.detect_level({"foo": "bar"})


class TestStructural:
    def test_text_level_ok(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(p, [_text_record("a", [1.0, 0.0]), _text_record("b", [0.0, 1.0])])
        r = v.validate_structural(p)
        assert r.passed
        assert r.records_checked == 2
        assert r.items_checked == 2

    def test_catches_dim_mismatch(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(p, [_text_record("a", [1.0, 0.0]), _text_record("b", [1.0])])
        r = v.validate_structural(p)
        assert not r.passed
        assert any("dim" in e for e in r.errors)

    def test_catches_nan(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(p, [_text_record("a", [float("nan"), 0.0])])
        r = v.validate_structural(p)
        assert not r.passed
        assert any("non-finite" in e for e in r.errors)

    def test_catches_bad_timestamp(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        rec = _text_record("a", [1.0, 0.0])
        rec["ts"] = "nope"
        _write_jsonl_bz2(p, [rec])
        r = v.validate_structural(p)
        assert not r.passed
        assert any("bad ts" in e for e in r.errors)

    def test_sentence_level_ok(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(
            p,
            [
                _sentence_record("c1", [(0, [1.0, 0.0]), (1, [0.0, 1.0])]),
                _sentence_record("c2", [(0, [0.5, 0.5])]),
            ],
        )
        r = v.validate_structural(p)
        assert r.passed
        assert r.records_checked == 2
        assert r.items_checked == 3

    def test_mixed_levels_in_file_fails(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(
            p,
            [
                _text_record("a", [1.0, 0.0]),
                _sentence_record("c1", [(0, [1.0, 0.0])]),
            ],
        )
        r = v.validate_structural(p)
        assert not r.passed
        assert any("differs from file-wide" in e for e in r.errors)

    def test_malformed_json_fails(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        with bz2.open(p, "wt") as fh:
            fh.write("not json\n")
        r = v.validate_structural(p)
        assert not r.passed


class TestComparison:
    def test_self_comparison_passes(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(p, [_text_record("a", [1.0, 0.0]), _text_record("b", [0.0, 1.0])])
        r = v.validate_against_target(p, p, tol=v.DEFAULT_TOL)
        assert r.passed
        assert r.records_checked == 2
        assert r.max_distance < 1e-12

    def test_detects_distance_above_tol(self, tmp_path):
        a = tmp_path / "a.jsonl.bz2"
        b = tmp_path / "b.jsonl.bz2"
        _write_jsonl_bz2(a, [_text_record("x", [1.0, 0.0])])
        _write_jsonl_bz2(b, [_text_record("x", [0.0, 1.0])])  # orthogonal → cos dist = 1
        r = v.validate_against_target(a, b, tol=0.5)
        assert not r.passed
        assert any("cosine distance" in m for m in r.mismatches)

    def test_detects_missing_record(self, tmp_path):
        a = tmp_path / "a.jsonl.bz2"
        b = tmp_path / "b.jsonl.bz2"
        _write_jsonl_bz2(a, [_text_record("x", [1.0, 0.0]), _text_record("y", [1.0, 0.0])])
        _write_jsonl_bz2(b, [_text_record("x", [1.0, 0.0])])
        r = v.validate_against_target(a, b)
        assert not r.passed
        assert any("missing in target" in m for m in r.mismatches)

    def test_sentence_level_self_comparison(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(
            p,
            [_sentence_record("c1", [(0, [1.0, 0.0]), (1, [0.0, 1.0])])],
        )
        r = v.validate_against_target(p, p)
        assert r.passed
        assert r.items_checked == 2

    def test_level_mismatch_fails(self, tmp_path):
        a = tmp_path / "a.jsonl.bz2"
        b = tmp_path / "b.jsonl.bz2"
        _write_jsonl_bz2(a, [_text_record("x", [1.0, 0.0])])
        _write_jsonl_bz2(b, [_sentence_record("x", [(0, [1.0, 0.0])])])
        r = v.validate_against_target(a, b)
        assert not r.passed
        assert any("level mismatch" in e for e in r.errors)

    def test_empty_files_both_pass(self, tmp_path):
        a = tmp_path / "a.jsonl.bz2"
        b = tmp_path / "b.jsonl.bz2"
        _write_jsonl_bz2(a, [])
        _write_jsonl_bz2(b, [])
        assert v.validate_against_target(a, b).passed

    def test_one_side_empty_fails(self, tmp_path):
        a = tmp_path / "a.jsonl.bz2"
        b = tmp_path / "b.jsonl.bz2"
        _write_jsonl_bz2(a, [_text_record("x", [1.0, 0.0])])
        _write_jsonl_bz2(b, [])
        r = v.validate_against_target(a, b)
        assert not r.passed


class TestIterLinesFromPath:
    def test_local_path(self, tmp_path):
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(p, [{"a": 1}, {"b": 2}])
        lines = list(v.iter_lines_from_path(p))
        assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]
