import json
import re

from impresso_text_embedder.schema import (
    ChunkItem,
    ChunkRecord,
    SentenceItem,
    SentenceRecord,
    TextRecord,
    utc_timestamp,
)


class TestUtcTimestamp:
    def test_format(self):
        ts = utc_timestamp()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts)


class TestTextRecord:
    def test_round_and_drop_none(self):
        r = TextRecord(
            ci_id="ci-1",
            model_id="foo@bar",
            embedding=[0.123456789, -0.987654321, 0.0],
            size=3,
            ts="2024-01-02T03:04:05Z",
        )
        d = r.to_dict()
        assert d == {
            "ci_id": "ci-1",
            "model_id": "foo@bar",
            "embedding": [0.12346, -0.98765, 0.0],
            "size": 3,
            "ts": "2024-01-02T03:04:05Z",
        }
        assert "ci_type" not in d  # None dropped
        # round-trip through json
        assert json.loads(json.dumps(d)) == d

    def test_ts_and_ci_type_optional(self):
        r = TextRecord(
            ci_id="x", model_id="e", embedding=[0.1], size=1
        )
        d = r.to_dict()
        assert "ts" not in d
        assert "ci_type" not in d

    def test_ci_type_included_when_set(self):
        r = TextRecord(
            ci_id="x", model_id="e", embedding=[0.1], size=1, ci_type="ar"
        )
        assert r.to_dict()["ci_type"] == "ar"


class TestSentenceRecord:
    def test_nested_items_rounded(self):
        rec = SentenceRecord(
            ts="t",
            ci_id="ci-1",
            sents=[
                SentenceItem(sent_id=0, embedding=[1.111111, 2.222222], size=2, lg="fr"),
                SentenceItem(sent_id=1, embedding=[3.333333], size=1, o=15),
            ],
        )
        d = rec.to_dict()
        assert d["ci_id"] == "ci-1"
        assert d["sents"] == [
            {"sent_id": 0, "embedding": [1.11111, 2.22222], "size": 2, "lg": "fr"},
            {"sent_id": 1, "embedding": [3.33333], "size": 1, "o": 15},
        ]
        for optional in ("model_id", "lingproc_path", "git"):
            assert optional not in d

    def test_optional_metadata_kept_when_set(self):
        rec = SentenceRecord(
            ts="t", ci_id="x", sents=[], model_id="m", lingproc_path="p", git="abc123"
        )
        d = rec.to_dict()
        assert d["model_id"] == "m"
        assert d["lingproc_path"] == "p"
        assert d["git"] == "abc123"


class TestChunkRecord:
    def test_shape_and_rounding(self):
        rec = ChunkRecord(
            ts="t",
            ci_id="ci-2",
            chunks=[ChunkItem(chunk_id=0, embedding=[0.000001, 0.999999], size=2)],
        )
        d = rec.to_dict()
        assert d["chunks"] == [
            {"chunk_id": 0, "embedding": [0.0, 1.0], "size": 2},
        ]
