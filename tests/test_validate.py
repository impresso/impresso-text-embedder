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


def _text_record(ci_id: str, emb: list[float]) -> dict:
    return {
        "ci_id": ci_id,
        "model_id": "m@default",
        "embedding": emb,
        "size": len(emb),
        "ts": "2024-01-02T03:04:05Z",
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
        # orjson is strict RFC 8259 and rejects bare ``NaN`` at parse time, so
        # a NaN in the file surfaces as a malformed-JSON error rather than as
        # a non-finite-embedding error. Either way the file is flagged.
        p = tmp_path / "e.jsonl.bz2"
        _write_jsonl_bz2(p, [_text_record("a", [float("nan"), 0.0])])
        r = v.validate_structural(p)
        assert not r.passed
        assert r.errors

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
        assert any("cosine distance" in str(m) for m in r.mismatches)
        assert any(m.kind == v.MismatchKind.VALUE for m in r.mismatches)

    def test_detects_missing_record(self, tmp_path):
        a = tmp_path / "a.jsonl.bz2"
        b = tmp_path / "b.jsonl.bz2"
        _write_jsonl_bz2(a, [_text_record("x", [1.0, 0.0]), _text_record("y", [1.0, 0.0])])
        _write_jsonl_bz2(b, [_text_record("x", [1.0, 0.0])])
        r = v.validate_against_target(a, b)
        assert not r.passed
        assert any("missing in target" in str(m) for m in r.mismatches)
        assert any(m.kind == v.MismatchKind.MISSING_IN_TARGET for m in r.mismatches)

    def test_distances_recorded_for_every_compared_pair(self, tmp_path):
        a = tmp_path / "a.jsonl.bz2"
        b = tmp_path / "b.jsonl.bz2"
        _write_jsonl_bz2(
            a,
            [
                _text_record("x", [1.0, 0.0]),
                _text_record("y", [1.0, 0.0]),
                _text_record("z", [1.0, 0.0]),
            ],
        )
        _write_jsonl_bz2(
            b,
            [
                _text_record("x", [1.0, 0.0]),       # 0
                _text_record("y", [0.999, 0.0449]),  # small drift
                _text_record("z", [0.0, 1.0]),       # large drift
            ],
        )
        r = v.validate_against_target(a, b, tol=1e-4)
        assert len(r.distances) == 3
        assert r.level == "text"
        # One VALUE mismatch per above-tol pair (here 2).
        vals = [m for m in r.mismatches if m.kind == v.MismatchKind.VALUE]
        assert len(vals) == 2
        # Worst drift is largest distance.
        assert max(m.distance for m in vals) == pytest.approx(max(r.distances))

    def test_mismatch_str_preserves_markers(self):
        m_val = v.Mismatch(v.MismatchKind.VALUE, ci_id="x", distance=2e-4, tol=1e-4)
        assert "cosine distance" in str(m_val)
        assert "tol 1e-04" in str(m_val)
        m_miss = v.Mismatch(v.MismatchKind.MISSING_IN_TARGET, ci_id="y")
        assert "missing in target" in str(m_miss)
        assert "ci_id='y'" in str(m_miss)
        m_item = v.Mismatch(
            v.MismatchKind.MISSING_IN_PRODUCED, ci_id="z", item_id=7, id_key="sent_id"
        )
        assert "missing in produced" in str(m_item)
        assert "sent_id=7" in str(m_item)

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


# --------------------------------------------------------------------------- #
# Source-stats
# --------------------------------------------------------------------------- #


def _source_record(
    rid: str,
    *,
    tp: str | None = None,
    lg: str | None = None,
    ft: str | None = None,
    sents: list[dict] | None = None,
) -> dict:
    out: dict = {"id": rid}
    if tp is not None:
        out["tp"] = tp
    if lg is not None:
        out["lg"] = lg
    if ft is not None:
        out["ft"] = ft
    if sents is not None:
        out["sents"] = sents
    return out


class TestSourceStats:
    def test_populates_both_directions(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        _write_jsonl_bz2(
            produced,
            [_text_record("p1", [1.0, 0.0]), _text_record("p3", [1.0, 0.0])],
        )
        _write_jsonl_bz2(
            target,
            [_text_record("p1", [1.0, 0.0]), _text_record("p2", [1.0, 0.0])],
        )
        # p3 is missing in target; p2 is missing in produced.
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(
            source,
            [
                _source_record("p2", tp="ar", lg="fr", ft="Alors " * 100),
                _source_record("p3"),  # empty: no ft, no sents
            ],
        )

        report = v.validate_against_target(produced, target)
        analysis = v.collect_source_stats(source, report, min_char_length=400)

        t_block = analysis.blocks[v.MismatchKind.MISSING_IN_TARGET]
        assert t_block.total == 1
        assert t_block.found_in_source == 1
        assert t_block.reconstructable == 0
        assert t_block.empty == 1

        p_block = analysis.blocks[v.MismatchKind.MISSING_IN_PRODUCED]
        assert p_block.total == 1
        assert p_block.found_in_source == 1
        assert p_block.reconstructable == 1
        assert p_block.empty == 0

    def test_attached_to_report(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        _write_jsonl_bz2(produced, [_text_record("x", [1.0, 0.0])])
        _write_jsonl_bz2(target, [])
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [_source_record("x", ft="hello")])

        report = v.validate_against_target(produced, target)
        assert report.source_stats is None
        analysis = v.collect_source_stats(source, report)
        assert report.source_stats is analysis

    def test_reconstructs_from_sents_when_ft_absent(self, tmp_path):
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(
            source,
            [
                _source_record(
                    "x",
                    tp="ar",
                    lg="de",
                    sents=[
                        {"tok": [{"t": "Hallo", "o": 0}, {"t": "Welt", "o": 6}]}
                    ],
                )
            ],
        )

        report = v.ValidationReport()
        report.mismatches.append(v.Mismatch(v.MismatchKind.MISSING_IN_TARGET, ci_id="x"))
        analysis = v.collect_source_stats(source, report, min_char_length=400)

        block = analysis.blocks[v.MismatchKind.MISSING_IN_TARGET]
        assert block.reconstructable == 1
        assert block.empty == 0
        # "Hallo" (5) + " " (1 pad from offset gap) + "Welt" (4) = 10.
        assert block.char_lengths == [10]
        assert block.lg_counts["de"] == 1
        assert block.tp_counts["ar"] == 1

    def test_tracks_not_in_source(self, tmp_path):
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [_source_record("other", ft="x")])

        report = v.ValidationReport()
        report.mismatches.append(v.Mismatch(v.MismatchKind.MISSING_IN_TARGET, ci_id="x"))
        report.mismatches.append(v.Mismatch(v.MismatchKind.MISSING_IN_PRODUCED, ci_id="y"))
        analysis = v.collect_source_stats(source, report)

        t_block = analysis.blocks[v.MismatchKind.MISSING_IN_TARGET]
        assert t_block.total == 1
        assert t_block.found_in_source == 0
        assert t_block.not_in_source_ids == ["x"]
        p_block = analysis.blocks[v.MismatchKind.MISSING_IN_PRODUCED]
        assert p_block.not_in_source_ids == ["y"]

    def test_missing_lg_tp_fall_into_missing_bucket(self, tmp_path):
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [_source_record("x", ft="hello")])

        report = v.ValidationReport()
        report.mismatches.append(v.Mismatch(v.MismatchKind.MISSING_IN_TARGET, ci_id="x"))
        analysis = v.collect_source_stats(source, report)

        block = analysis.blocks[v.MismatchKind.MISSING_IN_TARGET]
        assert block.lg_counts["(missing)"] == 1
        assert block.tp_counts["(missing)"] == 1

    def test_below_min_char_tally(self, tmp_path):
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(
            source,
            [
                _source_record("short", ft="abc"),
                _source_record("long", ft="x" * 20),
            ],
        )

        report = v.ValidationReport()
        for rid in ("short", "long"):
            report.mismatches.append(v.Mismatch(v.MismatchKind.MISSING_IN_TARGET, ci_id=rid))
        analysis = v.collect_source_stats(source, report, min_char_length=10)

        block = analysis.blocks[v.MismatchKind.MISSING_IN_TARGET]
        assert block.below_min_char == 1

    def test_samples_capped(self, tmp_path):
        source = tmp_path / "s.jsonl.bz2"
        records = [_source_record(f"x{i}", ft="content " * 20) for i in range(10)]
        _write_jsonl_bz2(source, records)

        report = v.ValidationReport()
        for i in range(10):
            report.mismatches.append(v.Mismatch(v.MismatchKind.MISSING_IN_TARGET, ci_id=f"x{i}"))
        analysis = v.collect_source_stats(source, report, sample_count=3, excerpt_chars=20)

        block = analysis.blocks[v.MismatchKind.MISSING_IN_TARGET]
        assert len(block.samples) == 3
        for sample in block.samples:
            assert isinstance(sample, v.Sample)
            assert sample.ci_id.startswith("x")
            assert len(sample.excerpt) <= 20
            # Missing-direction samples carry no distance.
            assert sample.distance is None

    def test_no_missing_ids_returns_empty_blocks(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        _write_jsonl_bz2(produced, [_text_record("x", [1.0, 0.0])])
        _write_jsonl_bz2(target, [_text_record("x", [1.0, 0.0])])
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [])

        report = v.validate_against_target(produced, target)
        analysis = v.collect_source_stats(source, report)
        for block in analysis.blocks.values():
            assert block.total == 0
            assert block.found_in_source == 0

    def test_item_level_mismatches_collapse_to_record(self, tmp_path):
        # One ci_id with several sent-level mismatches should count as 1, not N.
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [_source_record("c1", ft="some content")])

        report = v.ValidationReport()
        for sent_id in range(3):
            report.mismatches.append(
                v.Mismatch(
                    v.MismatchKind.MISSING_IN_TARGET,
                    ci_id="c1",
                    item_id=sent_id,
                    id_key="sent_id",
                )
            )
        analysis = v.collect_source_stats(source, report)
        block = analysis.blocks[v.MismatchKind.MISSING_IN_TARGET]
        assert block.total == 1
        assert block.found_in_source == 1


class TestSourceStatsValueDirection:
    def test_value_direction_populated(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        # One above-tol record.
        _write_jsonl_bz2(produced, [_text_record("d1", [1.0, 0.0])])
        _write_jsonl_bz2(target, [_text_record("d1", [0.0, 1.0])])
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [_source_record("d1", tp="ar", lg="fr", ft="texte " * 50)])

        report = v.validate_against_target(produced, target, tol=1e-4)
        analysis = v.collect_source_stats(source, report)

        block = analysis.blocks[v.MismatchKind.VALUE]
        assert block.total == 1
        assert block.found_in_source == 1
        assert block.reconstructable == 1
        assert len(block.samples) == 1
        s = block.samples[0]
        assert s.ci_id == "d1"
        assert s.distance is not None and s.distance > 0.9

    def test_value_samples_ordered_by_worst_drift(self, tmp_path):
        # Craft four above-tol records with monotonically-increasing drift.
        # Orthogonal-like pairs produce distances from small → near 1.0
        # depending on the second component magnitude.
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        # For deterministic ordering we hand-craft embeddings whose cosine
        # distances sort as d1 < d2 < d3 < d4.
        p_recs = [
            _text_record("d1", [1.0, 0.0]),
            _text_record("d2", [1.0, 0.0]),
            _text_record("d3", [1.0, 0.0]),
            _text_record("d4", [1.0, 0.0]),
        ]
        t_recs = [
            _text_record("d1", [1.0, 0.05]),   # small drift
            _text_record("d2", [1.0, 0.30]),   # larger
            _text_record("d3", [1.0, 1.0]),    # 45°
            _text_record("d4", [-1.0, 0.0]),   # antiparallel → 2.0
        ]
        _write_jsonl_bz2(produced, p_recs)
        _write_jsonl_bz2(target, t_recs)

        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(
            source,
            [_source_record(f"d{i}", tp="ar", lg="fr", ft=f"body {i} " * 20) for i in range(1, 5)],
        )

        report = v.validate_against_target(produced, target, tol=1e-4)
        analysis = v.collect_source_stats(source, report, sample_count=3)

        block = analysis.blocks[v.MismatchKind.VALUE]
        # Top 3 by descending distance → d4, d3, d2.
        assert [s.ci_id for s in block.samples] == ["d4", "d3", "d2"]
        dists = [s.distance for s in block.samples]
        assert dists[0] > dists[1] > dists[2]

    def test_value_items_collapse_to_record_max_distance(self, tmp_path):
        # Synthesise three item-level VALUE mismatches for one ci_id with
        # distinct distances; the ranking key should use the max.
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [_source_record("c1", ft="body " * 50)])

        report = v.ValidationReport()
        for sent_id, d in enumerate([0.01, 0.5, 0.02]):
            report.mismatches.append(
                v.Mismatch(
                    v.MismatchKind.VALUE,
                    ci_id="c1",
                    item_id=sent_id,
                    id_key="sent_id",
                    distance=d,
                    tol=1e-4,
                )
            )

        analysis = v.collect_source_stats(source, report)
        block = analysis.blocks[v.MismatchKind.VALUE]
        assert block.total == 1
        assert len(block.samples) == 1
        assert block.samples[0].distance == pytest.approx(0.5)

    def test_value_block_empty_when_no_drift(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        _write_jsonl_bz2(produced, [_text_record("x", [1.0, 0.0])])
        _write_jsonl_bz2(target, [_text_record("x", [1.0, 0.0])])
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(source, [_source_record("x", ft="hi")])

        report = v.validate_against_target(produced, target)
        analysis = v.collect_source_stats(source, report)
        assert analysis.blocks[v.MismatchKind.VALUE].total == 0
        assert analysis.blocks[v.MismatchKind.VALUE].found_in_source == 0

    def test_value_distances_helper_returns_max_per_record(self):
        ms = [
            v.Mismatch(v.MismatchKind.VALUE, ci_id="a", distance=0.1, tol=1e-4),
            v.Mismatch(v.MismatchKind.VALUE, ci_id="a", distance=0.9, tol=1e-4),
            v.Mismatch(v.MismatchKind.VALUE, ci_id="b", distance=0.3, tol=1e-4),
            v.Mismatch(v.MismatchKind.MISSING_IN_TARGET, ci_id="c"),  # ignored
        ]
        out = v._value_distance_per_record(ms)
        assert out == {"a": 0.9, "b": 0.3}


class TestCharLengthStats:
    def test_add_tracks_min_mean_max(self):
        s = v.CharLengthStats()
        for length in (300, 100, 500, 200):
            s.add(length)
        assert s.count == 4
        assert s.min == 100
        assert s.max == 500
        assert s.mean == pytest.approx(275.0)

    def test_empty_returns_none_mean(self):
        s = v.CharLengthStats()
        assert s.count == 0
        assert s.min is None
        assert s.max is None
        assert s.mean is None


class TestComparedCiIds:
    def test_text_level_populated_only_for_matched(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        _write_jsonl_bz2(
            produced,
            [_text_record("a", [1.0, 0.0]), _text_record("b", [1.0, 0.0])],
        )
        _write_jsonl_bz2(
            target,
            [_text_record("a", [1.0, 0.0]), _text_record("c", [1.0, 0.0])],
        )
        report = v.validate_against_target(produced, target)
        # Only "a" exists on both sides; b and c go into missing buckets.
        assert report.compared_ci_ids == {"a"}

    def test_sentence_level_records_compared_only(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        _write_jsonl_bz2(
            produced,
            [
                _sentence_record("ci-1", [(1, [1.0, 0.0]), (2, [1.0, 0.0])]),
                _sentence_record("ci-2", [(1, [1.0, 0.0])]),
            ],
        )
        _write_jsonl_bz2(
            target,
            [
                _sentence_record("ci-1", [(1, [1.0, 0.0]), (2, [1.0, 0.0])]),
            ],
        )
        report = v.validate_against_target(produced, target)
        assert report.compared_ci_ids == {"ci-1"}


class TestKeptBaseline:
    def test_baseline_tallied_for_kept_records(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        # k1, k2 match exactly (kept). m1 is in produced only (missing in target).
        _write_jsonl_bz2(
            produced,
            [
                _text_record("k1", [1.0, 0.0]),
                _text_record("k2", [1.0, 0.0]),
                _text_record("m1", [1.0, 0.0]),
            ],
        )
        _write_jsonl_bz2(
            target,
            [_text_record("k1", [1.0, 0.0]), _text_record("k2", [1.0, 0.0])],
        )
        # Source contains every record + an extra one filtered upstream.
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(
            source,
            [
                _source_record("k1", ft="x" * 1000),
                _source_record("k2", ft="x" * 200),
                _source_record("m1", ft="x" * 50),
                _source_record("filtered", ft="x" * 9999),  # not in compared
            ],
        )

        report = v.validate_against_target(produced, target)
        analysis = v.collect_source_stats(source, report)

        # Baseline contains exactly the two kept records — not m1, not the
        # filtered record.
        assert analysis.baseline.count == 2
        assert analysis.baseline.min == 200
        assert analysis.baseline.max == 1000
        assert analysis.baseline.mean == pytest.approx(600.0)

    def test_baseline_empty_when_no_kept_records(self, tmp_path):
        produced = tmp_path / "p.jsonl.bz2"
        target = tmp_path / "t.jsonl.bz2"
        _write_jsonl_bz2(produced, [_text_record("a", [1.0, 0.0])])
        _write_jsonl_bz2(target, [_text_record("b", [1.0, 0.0])])
        source = tmp_path / "s.jsonl.bz2"
        _write_jsonl_bz2(
            source,
            [_source_record("a", ft="x" * 50), _source_record("b", ft="x" * 50)],
        )

        report = v.validate_against_target(produced, target)
        analysis = v.collect_source_stats(source, report)
        # Both records are mismatches (missing on one side), neither kept.
        assert analysis.baseline.count == 0
        assert analysis.baseline.mean is None
