from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from impresso_text_embedder import embed as em
from impresso_text_embedder.chunking.base import Chunk, ChunkingStrategy


def _cfg(**over) -> em.EncoderConfig:
    defaults = dict(
        batch_size=3,
        normalize_embeddings=False,
        min_char_length=5,
        include_text=False,
        content_types=frozenset({"ar"}),
    )
    defaults.update(over)
    return em.EncoderConfig(**defaults)


def _record(
    ci_id: str = "ci-1",
    text: str = "Hello world, this is a long enough article body.",
    tp: str = "ar",
    lg: str | None = "fr",
    with_sents: bool = True,
) -> dict:
    rec: dict = {"id": ci_id, "tp": tp, "lg": lg, "lingproc_path": "some/path.jsonl.bz2"}
    if with_sents:
        # one sentence with the whole text as a single "token" at offset 0
        rec["sents"] = [{"tok": [{"t": text, "o": 0}]}]
    return rec


def _fake_encode(factory):
    """Patch encode_texts to return whatever the factory builds for given texts."""
    return lambda model, texts, **_: np.asarray(factory(texts), dtype=np.float32)


def test_build_embedder_tag():
    assert em.build_embedder_tag("foo/bar", None) == "foo/bar@default"
    assert em.build_embedder_tag("foo/bar", "abc") == "foo/bar@abc"


class TestTextBatcher:
    def _patch_encode(self, monkeypatch, dim: int = 2):
        def fake(texts):
            # embed_i = [i, len(texts[i])]; deterministic and small
            return [[float(i), float(len(t))] for i, t in enumerate(texts)]

        monkeypatch.setattr(em, "encode_texts", _fake_encode(fake))
        return dim

    def test_buffers_then_flushes_at_batch_size(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=2)
        model = MagicMock()
        b = em.TextBatcher(model, cfg, embedder_tag="m@r")
        out1 = b.add(_record("a"))
        out2 = b.add(_record("b"))  # second add triggers flush
        assert out1 == []
        assert [r.id for r in out2] == ["a", "b"]
        assert all(r.embedder == "m@r" for r in out2)
        # nothing more to flush after
        assert b.flush() == []

    def test_flush_empties_buffer_and_is_idempotent(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="tag")
        b.add(_record("a"))
        got = b.flush()
        assert [r.id for r in got] == ["a"]
        assert b.flush() == []

    def test_filters_wrong_content_type(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(content_types=frozenset({"ar"}))
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        assert b.add(_record("p", tp="page")) == []
        assert len(b) == 0

    def test_filters_short_text(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10, min_char_length=1000)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        assert b.add(_record("a", text="too short")) == []
        assert len(b) == 0

    def test_include_text_flag_controls_text_field(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10, include_text=True)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        b.add(_record("a"))
        [rec] = b.flush()
        d = rec.to_dict()
        assert "text" in d

    def test_skips_record_without_id(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        assert b.add(_record("", text="ok long enough string here")) == []


class TestEmbedSentenceRecord:
    def test_builds_record_from_sents(self, monkeypatch):
        monkeypatch.setattr(
            em, "encode_texts", _fake_encode(lambda texts: [[0.5, 0.5] for _ in texts])
        )
        rec = {
            "id": "ci-1",
            "tp": "ar",
            "lg": "fr",
            "sents": [
                {"tok": [{"t": "This sentence is long enough.", "o": 0}], "o": 0},
                {"tok": [{"t": "short", "o": 40}], "o": 40},  # short → filtered
                {"tok": [{"t": "Another long sentence here.", "o": 50}], "o": 50},
            ],
        }
        out = em.embed_sentence_record(rec, MagicMock(), _cfg(min_char_length=5))
        assert out is not None
        assert [s.sent_id for s in out.sents] == [0, 2]
        assert all(s.lg == "fr" for s in out.sents)
        assert out.lingproc_path is None  # not in record

    def test_returns_none_when_no_sents(self):
        out = em.embed_sentence_record({"id": "x", "tp": "ar"}, MagicMock(), _cfg())
        assert out is None

    def test_filters_by_content_type(self):
        out = em.embed_sentence_record(_record(tp="page"), MagicMock(), _cfg())
        assert out is None


class TestEmbedChunkRecord:
    class _FakeChunker(ChunkingStrategy):
        def __init__(self, chunks):
            self._chunks = chunks

        def chunk(self, text):
            return list(self._chunks)

    def test_builds_record_from_chunks(self, monkeypatch):
        monkeypatch.setattr(
            em, "encode_texts", _fake_encode(lambda texts: [[float(i)] for i in range(len(texts))])
        )
        chunker = self._FakeChunker([Chunk("chunk 1", 0), Chunk("chunk 2", 10)])
        out = em.embed_chunk_record(_record("ci-7"), MagicMock(), _cfg(), chunker)
        assert out is not None
        assert out.ci_id == "ci-7"
        assert [c.chunk_id for c in out.chunks] == [0, 1]
        assert [c.o for c in out.chunks] == [0, 10]

    def test_returns_none_when_chunker_returns_nothing(self, monkeypatch):
        monkeypatch.setattr(em, "encode_texts", MagicMock())
        chunker = self._FakeChunker([])
        assert em.embed_chunk_record(_record(), MagicMock(), _cfg(), chunker) is None


class TestEmbedRecords:
    def test_text_level_batches_across_records(self, monkeypatch):
        monkeypatch.setattr(
            em, "encode_texts", _fake_encode(lambda texts: [[0.0] for _ in texts])
        )
        records = [_record(f"c-{i}") for i in range(5)]
        out = list(
            em.embed_records(
                records, level="text", model=MagicMock(), cfg=_cfg(batch_size=2), embedder_tag="t"
            )
        )
        assert [o["id"] for o in out] == [f"c-{i}" for i in range(5)]

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="unknown embedding level"):
            list(em.embed_records([], level="bogus", model=MagicMock(), cfg=_cfg(), embedder_tag="t"))

    def test_chunk_level_requires_chunker(self):
        with pytest.raises(ValueError, match="chunking strategy"):
            list(em.embed_records([], level="chunk", model=MagicMock(), cfg=_cfg(), embedder_tag="t"))
