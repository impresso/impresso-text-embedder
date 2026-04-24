from __future__ import annotations

import logging
from collections import Counter
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
        assert [r.ci_id for r in out2] == ["a", "b"]
        assert all(r.model_id == "m@r" for r in out2)
        assert all(r.size == len(r.embedding) for r in out2)
        # nothing more to flush after
        assert b.flush() == []

    def test_flush_empties_buffer_and_is_idempotent(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="tag")
        b.add(_record("a"))
        got = b.flush()
        assert [r.ci_id for r in got] == ["a"]
        assert b.flush() == []

    def test_filters_wrong_content_type(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(content_types=frozenset({"ar"}))
        counter: Counter[str] = Counter()
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t", filter_counter=counter)
        assert b.add(_record("p", tp="page")) == []
        assert len(b) == 0
        assert counter[em.FILTER_CONTENT_TYPE] == 1
        assert counter[em.FILTER_MISSING_CONTENT_TYPE] == 0

    def test_filters_missing_content_type(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(content_types=frozenset({"ar"}))
        counter: Counter[str] = Counter()
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t", filter_counter=counter)
        rec_none = _record("a")
        rec_none["tp"] = None  # explicit None
        rec_absent = _record("b")
        del rec_absent["tp"]  # key absent
        assert b.add(rec_none) == []
        assert b.add(rec_absent) == []
        assert len(b) == 0
        assert counter[em.FILTER_MISSING_CONTENT_TYPE] == 2
        assert counter[em.FILTER_CONTENT_TYPE] == 0

    def test_warns_once_for_missing_content_type(self, monkeypatch, caplog):
        self._patch_encode(monkeypatch)
        cfg = _cfg(content_types=frozenset({"ar"}))
        counter: Counter[str] = Counter()
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t", filter_counter=counter)
        with caplog.at_level(logging.WARNING, logger="impresso_text_embedder.embed"):
            for ci in ("a", "b", "c"):
                r = _record(ci)
                del r["tp"]
                b.add(r)
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "impresso_text_embedder.embed"
        ]
        assert len(warnings) == 1
        assert "missing_content_type" in warnings[0].getMessage()
        assert counter[em.FILTER_MISSING_CONTENT_TYPE] == 3

    def test_wrong_content_type_does_not_warn(self, monkeypatch, caplog):
        self._patch_encode(monkeypatch)
        cfg = _cfg(content_types=frozenset({"ar"}))
        counter: Counter[str] = Counter()
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t", filter_counter=counter)
        with caplog.at_level(logging.WARNING, logger="impresso_text_embedder.embed"):
            b.add(_record("p", tp="page"))
            b.add(_record("q", tp="image"))
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "impresso_text_embedder.embed"
        ]
        assert warnings == []

    def test_filters_short_text(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10, min_char_length=1000)
        counter: Counter[str] = Counter()
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t", filter_counter=counter)
        assert b.add(_record("a", text="too short")) == []
        assert len(b) == 0
        assert counter[em.FILTER_TOO_SHORT] == 1

    def test_ci_type_echoed_from_input_tp(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        b.add(_record("a", tp="ar"))
        [rec] = b.flush()
        assert rec.ci_type == "ar"

    def test_skips_record_without_id(self, monkeypatch):
        self._patch_encode(monkeypatch)
        cfg = _cfg(batch_size=10)
        counter: Counter[str] = Counter()
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t", filter_counter=counter)
        assert b.add(_record("", text="ok long enough string here")) == []
        assert counter[em.FILTER_MISSING_ID] == 1

    def test_counter_optional(self, monkeypatch):
        # No counter passed → no crash; existing tests above rely on this.
        self._patch_encode(monkeypatch)
        b = em.TextBatcher(MagicMock(), _cfg(), embedder_tag="t")
        assert b.add(_record("p", tp="page")) == []


class TestTextBatcherLongDoc:
    """Long-document (>max_seq_length at text level) branch of TextBatcher.

    Exercises the chunk+aggregate path without loading a real model or
    tokenizer. The token counter is a stub keyed off a per-test token
    map; the chunker returns caller-supplied chunks; the aggregator
    returns a constant vector. This keeps the unit tests fast and
    independent of xformers / CUDA / HF.
    """

    def _make_long_doc_cfg(
        self,
        *,
        chunks_for: dict[str, list[str]] | None = None,
        token_counts: dict[str, int] | None = None,
        model_max_tokens: int = 100,
    ):
        """Build a LongDocConfig with fake chunker, aggregator, counter.

        ``chunks_for`` maps a document text → list of chunk strings the
        fake chunker returns for that text. ``token_counts`` maps text →
        reported token count. Returns (cfg_overrides, agg_mock).
        """
        from impresso_text_embedder.aggregation.base import AggregationStrategy
        from impresso_text_embedder.chunking.base import Chunk as _Chunk
        from impresso_text_embedder.chunking.base import (
            ChunkingStrategy as _ChunkingStrategy,
        )
        from impresso_text_embedder.embed import LongDocConfig

        counts = dict(token_counts or {})
        chunks_map = dict(chunks_for or {})

        def counter(text: str) -> int:
            return counts.get(text, len(text.split()))

        class _MapChunker(_ChunkingStrategy):
            def chunk(self, text):  # type: ignore[override]
                pieces = chunks_map.get(text)
                if pieces is None:
                    return []
                return [_Chunk(text=p, start=None) for p in pieces]

        # Aggregator records its inputs so tests can assert on them;
        # returns a recognisable constant vector.
        class _Agg(AggregationStrategy):
            def __init__(self):
                self.calls: list[np.ndarray] = []

            def aggregate(self, vectors, weights=None):  # type: ignore[override]
                self.calls.append(vectors.copy())
                out = np.zeros(vectors.shape[1], dtype=np.float32)
                out[0] = 1.0
                return out

        agg = _Agg()
        long_doc = LongDocConfig(
            strategy="chunk",
            chunker=_MapChunker(),
            aggregator=agg,
            model_max_tokens=model_max_tokens,
            token_counter=counter,
            # Disable the cheap char-estimate gate for tests — we want the
            # token counter to always run so we can control detection.
            char_fast_estimate=0.001,
        )
        return long_doc, agg

    def _patch_encode_identity(self, monkeypatch, dim: int = 2):
        """Encoder returns len(texts) vectors, each [idx, len(text)]."""

        def fake(texts):
            return [[float(i), float(len(t))] for i, t in enumerate(texts)]

        monkeypatch.setattr(em, "encode_texts", _fake_encode(fake))
        return dim

    def test_short_doc_bypasses_long_path(self, monkeypatch):
        self._patch_encode_identity(monkeypatch)
        long_doc, agg = self._make_long_doc_cfg(
            token_counts={"short text here.": 3},
            model_max_tokens=100,
        )
        cfg = _cfg(batch_size=1, long_doc=long_doc, min_char_length=3)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        [out] = b.add(_record("a", text="short text here."))
        # Aggregator was never called because the doc was short.
        assert agg.calls == []
        # Size matches the single-vector dim.
        assert out.size == len(out.embedding)

    def test_long_doc_chunks_and_aggregates(self, monkeypatch):
        self._patch_encode_identity(monkeypatch)
        long_text = "this is a pretty long document " * 10  # 60 "tokens" by split
        long_doc, agg = self._make_long_doc_cfg(
            chunks_for={long_text: ["c1", "c2", "c3"]},
            token_counts={long_text: 200},  # >= model_max_tokens=100
            model_max_tokens=100,
        )
        cfg = _cfg(batch_size=10, long_doc=long_doc, min_char_length=3)
        counter: Counter[str] = Counter()
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t", filter_counter=counter)
        b.add(_record("a", text=long_text))
        [out] = b.flush()
        assert out.ci_id == "a"
        # Aggregator called once with a [3, D] array (3 chunks).
        assert len(agg.calls) == 1
        assert agg.calls[0].shape[0] == 3
        # Fake aggregator always returns [1, 0] → emitted as embedding.
        assert out.embedding[0] == 1.0
        # Telemetry counter bumped.
        assert counter[em.LONG_DOC_CHUNKED] == 1

    def test_strategy_truncate_preserves_legacy_path(self, monkeypatch):
        """``strategy='truncate'`` means the detect/chunk/aggregate pipeline is inactive.

        Even with a giant token count reported, the batcher should send one
        text to the encoder and never call the aggregator.
        """
        self._patch_encode_identity(monkeypatch)
        from impresso_text_embedder.embed import LongDocConfig

        long_doc = LongDocConfig(
            strategy="truncate",
            chunker=None,
            aggregator=None,
            model_max_tokens=10,
            token_counter=lambda t: 999,  # always report "long"
        )
        cfg = _cfg(batch_size=1, long_doc=long_doc, min_char_length=3)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        [out] = b.add(_record("a", text="would be long if strategy were chunk"))
        assert out.size == len(out.embedding)

    def test_long_and_short_coexist_in_one_flush(self, monkeypatch):
        """A long doc's K chunks ride the same encode batch as short docs."""
        encoded: list[list[str]] = []

        def fake(texts):
            encoded.append(list(texts))
            return [[float(i), float(len(t))] for i, t in enumerate(texts)]

        monkeypatch.setattr(em, "encode_texts", _fake_encode(fake))

        long_text = "long doc body. " * 20
        long_doc, agg = self._make_long_doc_cfg(
            chunks_for={long_text: ["lc1", "lc2"]},
            token_counts={
                long_text: 999,
                "short one here.": 3,
                "second short.": 3,
            },
            model_max_tokens=50,
        )
        cfg = _cfg(batch_size=10, long_doc=long_doc, min_char_length=3)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        b.add(_record("short-a", text="short one here."))
        b.add(_record("long-a", text=long_text))
        b.add(_record("short-b", text="second short."))
        out = b.flush()
        assert [r.ci_id for r in out] == ["short-a", "long-a", "short-b"]
        # One encode call with all four texts (1 + 2 + 1).
        assert len(encoded) == 1
        assert len(encoded[0]) == 4
        # Aggregator was called exactly once (for the long doc).
        assert len(agg.calls) == 1

    def test_chunker_returns_empty_falls_back_to_one_shot(self, monkeypatch, caplog):
        self._patch_encode_identity(monkeypatch)
        long_text = "pathological input"
        long_doc, agg = self._make_long_doc_cfg(
            chunks_for={long_text: []},  # chunker returns [] for this text
            token_counts={long_text: 999},
            model_max_tokens=10,
        )
        cfg = _cfg(batch_size=10, long_doc=long_doc, min_char_length=3)
        b = em.TextBatcher(MagicMock(), cfg, embedder_tag="t")
        with caplog.at_level("WARNING"):
            b.add(_record("a", text=long_text))
        [out] = b.flush()
        # No aggregation — fallback to single-text path.
        assert agg.calls == []
        assert out.size == len(out.embedding)
        assert any("falling back to one-shot" in r.message for r in caplog.records)


class TestIsLongDoc:
    def test_returns_false_without_token_counter(self):
        from impresso_text_embedder.embed import LongDocConfig, is_long_doc

        cfg = LongDocConfig(strategy="chunk", model_max_tokens=10, token_counter=None)
        assert is_long_doc("a" * 1000, cfg) is False

    def test_char_estimate_short_circuits(self):
        from impresso_text_embedder.embed import LongDocConfig, is_long_doc

        called = {"n": 0}

        def counter(text: str) -> int:
            called["n"] += 1
            return 9999

        # char_fast_estimate=3.0 means "≥ 3 chars per token";
        # a 10-char input maps to at most ~3 tokens, well below 100.
        cfg = LongDocConfig(
            strategy="chunk",
            model_max_tokens=100,
            token_counter=counter,
            char_fast_estimate=3.0,
        )
        assert is_long_doc("short text", cfg) is False
        assert called["n"] == 0

    def test_long_text_triggers_counter(self):
        from impresso_text_embedder.embed import LongDocConfig, is_long_doc

        cfg = LongDocConfig(
            strategy="chunk",
            model_max_tokens=10,
            token_counter=lambda t: 500,
            char_fast_estimate=1.0,
        )
        assert is_long_doc("x" * 50, cfg) is True


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
        counter: Counter[str] = Counter()
        out = em.embed_sentence_record(
            {"id": "x", "tp": "ar"}, MagicMock(), _cfg(), filter_counter=counter
        )
        assert out is None
        assert counter[em.FILTER_NO_SENTENCES] == 1

    def test_filters_by_content_type(self):
        counter: Counter[str] = Counter()
        out = em.embed_sentence_record(
            _record(tp="page"), MagicMock(), _cfg(), filter_counter=counter
        )
        assert out is None
        assert counter[em.FILTER_CONTENT_TYPE] == 1
        assert counter[em.FILTER_MISSING_CONTENT_TYPE] == 0

    def test_filters_missing_content_type(self):
        counter: Counter[str] = Counter()
        rec = _record()
        del rec["tp"]
        out = em.embed_sentence_record(
            rec, MagicMock(), _cfg(), filter_counter=counter
        )
        assert out is None
        assert counter[em.FILTER_MISSING_CONTENT_TYPE] == 1
        assert counter[em.FILTER_CONTENT_TYPE] == 0

    def test_filters_when_all_sentences_short(self):
        counter: Counter[str] = Counter()
        rec = {
            "id": "ci-1",
            "tp": "ar",
            "sents": [
                {"tok": [{"t": "tiny", "o": 0}], "o": 0},
                {"tok": [{"t": "also", "o": 10}], "o": 10},
            ],
        }
        out = em.embed_sentence_record(
            rec, MagicMock(), _cfg(min_char_length=10), filter_counter=counter
        )
        assert out is None
        assert counter[em.FILTER_SENTENCE_TOO_SHORT] == 1


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
        counter: Counter[str] = Counter()
        assert (
            em.embed_chunk_record(
                _record(), MagicMock(), _cfg(), chunker, filter_counter=counter
            )
            is None
        )
        assert counter[em.FILTER_NO_CHUNKS] == 1

    def test_filters_missing_content_type(self):
        chunker = self._FakeChunker([Chunk("anything", 0)])
        counter: Counter[str] = Counter()
        rec = _record()
        del rec["tp"]
        out = em.embed_chunk_record(
            rec, MagicMock(), _cfg(), chunker, filter_counter=counter
        )
        assert out is None
        assert counter[em.FILTER_MISSING_CONTENT_TYPE] == 1
        assert counter[em.FILTER_CONTENT_TYPE] == 0


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
        assert [o["ci_id"] for o in out] == [f"c-{i}" for i in range(5)]

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="unknown embedding level"):
            list(em.embed_records([], level="bogus", model=MagicMock(), cfg=_cfg(), embedder_tag="t"))

    def test_chunk_level_requires_chunker(self):
        with pytest.raises(ValueError, match="chunking strategy"):
            list(em.embed_records([], level="chunk", model=MagicMock(), cfg=_cfg(), embedder_tag="t"))
