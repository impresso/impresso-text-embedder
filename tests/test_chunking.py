from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from impresso_text_embedder import chunking
from impresso_text_embedder.chunking.base import (
    _FACTORIES,
    Chunk,
    ChunkingStrategy,
    register_strategy,
)


class _FixedStrategy(ChunkingStrategy):
    def __init__(self, out):
        self._out = out

    def chunk(self, text: str):
        return list(self._out)


class TestRegistry:
    def test_semantic_is_registered_by_default(self):
        assert "semantic" in chunking.available_strategies()

    def test_register_and_get(self):
        token = object()
        registered = _FixedStrategy([Chunk(text="x")])

        def factory():
            # identity check — same instance is returned
            return registered

        register_strategy("fake-test", factory)
        try:
            got = chunking.get_strategy("fake-test")
            assert got is registered
        finally:
            _FACTORIES.pop("fake-test", None)
        assert token  # silence

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown chunking strategy"):
            chunking.get_strategy("does-not-exist")

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError):
            register_strategy("", lambda: _FixedStrategy([]))

    def test_available_strategies_is_sorted(self):
        listed = chunking.available_strategies()
        assert listed == sorted(listed)

    def test_get_strategy_builds_fresh_instance_per_call(self):
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return _FixedStrategy([])

        register_strategy("fresh-test", factory)
        try:
            chunking.get_strategy("fresh-test")
            chunking.get_strategy("fresh-test")
        finally:
            _FACTORIES.pop("fresh-test", None)
        assert calls["n"] == 2


class TestSemanticStrategy:
    def _make(self, raw_chunks):
        fake_chunker = MagicMock()
        fake_chunker.chunk.return_value = raw_chunks
        fake_ctor = MagicMock(return_value=fake_chunker)

        import impresso_text_embedder.chunking.semantic as semantic_mod

        with patch.dict("sys.modules", {"chonkie": MagicMock(SemanticChunker=fake_ctor)}):
            strat = semantic_mod.SemanticStrategy()
        return strat, fake_chunker, fake_ctor

    def test_constructor_forwards_defaults(self):
        _, _, fake_ctor = self._make([])
        fake_ctor.assert_called_once_with(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=1024,
            min_sentences=5,
        )

    def test_empty_text_short_circuits_without_calling_chunker(self):
        strat, fake_chunker, _ = self._make([])
        assert strat.chunk("") == []
        fake_chunker.chunk.assert_not_called()

    def test_maps_raw_chunks_with_start_attr(self):
        raw = [
            SimpleNamespace(text="hello", start=0),
            SimpleNamespace(text="world", start=6),
        ]
        strat, _, _ = self._make(raw)
        got = strat.chunk("hello world")
        assert got == [Chunk("hello", 0), Chunk("world", 6)]

    def test_falls_back_to_start_char(self):
        raw = [SimpleNamespace(text="x", start_char=42)]
        # start_char is read only when `start` is missing
        strat, _, _ = self._make(raw)
        [c] = strat.chunk("x")
        assert c.start == 42

    def test_start_defaults_to_none_when_absent(self):
        raw = [SimpleNamespace(text="only")]
        strat, _, _ = self._make(raw)
        [c] = strat.chunk("only")
        assert c.start is None

    def test_handles_none_return_from_chunker(self):
        strat, fake_chunker, _ = self._make(None)
        assert strat.chunk("something") == []
        fake_chunker.chunk.assert_called_once_with("something")
