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


def _make_fake_chunker_cls(*, kwarg_name: str, tokenizer_is_property: bool, chunks):
    """Build a real class mimicking a chonkie SemanticChunker variant.

    SemanticStrategy uses ``inspect.signature(SemanticChunker.__init__)``
    to pick the right kwarg, and tries to set ``chunker.tokenizer`` /
    ``chunker._tokenizer`` depending on chonkie version. A real class
    is needed (not MagicMock) so signature inspection and the property
    semantics behave the same way they do in the real chonkie.
    """
    if kwarg_name == "min_sentences_per_chunk":
        if tokenizer_is_property:
            class _FakeNewWithProperty:
                def __init__(
                    self,
                    embedding_model,
                    threshold,
                    chunk_size,
                    min_sentences_per_chunk,
                    **kw,
                ):
                    self._init_kwargs = {
                        "embedding_model": embedding_model,
                        "threshold": threshold,
                        "chunk_size": chunk_size,
                        "min_sentences_per_chunk": min_sentences_per_chunk,
                    }
                    self._tokenizer = None

                @property
                def tokenizer(self):
                    return self._tokenizer

                def chunk(self, text):
                    self._chunked_text = text
                    return chunks

            return _FakeNewWithProperty

        class _FakeNew:
            def __init__(
                self,
                embedding_model,
                threshold,
                chunk_size,
                min_sentences_per_chunk,
                **kw,
            ):
                self._init_kwargs = {
                    "embedding_model": embedding_model,
                    "threshold": threshold,
                    "chunk_size": chunk_size,
                    "min_sentences_per_chunk": min_sentences_per_chunk,
                }
                self.tokenizer = None

            def chunk(self, text):
                self._chunked_text = text
                return chunks

        return _FakeNew

    if kwarg_name == "min_sentences":
        class _FakeOld:
            def __init__(
                self, embedding_model, threshold, chunk_size, min_sentences, **kw
            ):
                self._init_kwargs = {
                    "embedding_model": embedding_model,
                    "threshold": threshold,
                    "chunk_size": chunk_size,
                    "min_sentences": min_sentences,
                }
                self.tokenizer = None

            def chunk(self, text):
                self._chunked_text = text
                return chunks

        return _FakeOld

    raise AssertionError(f"unhandled kwarg_name={kwarg_name!r}")


class TestSemanticStrategy:
    def _make(self, raw_chunks):
        cls = _make_fake_chunker_cls(
            kwarg_name="min_sentences_per_chunk",
            tokenizer_is_property=True,
            chunks=raw_chunks,
        )
        import impresso_text_embedder.chunking.semantic as semantic_mod

        with patch.dict("sys.modules", {"chonkie": MagicMock(SemanticChunker=cls)}):
            strat = semantic_mod.SemanticStrategy()
        return strat, strat._chunker, cls

    def test_constructor_forwards_defaults(self):
        _, chunker, _ = self._make([])
        assert chunker._init_kwargs == {
            "embedding_model": "minishlab/potion-base-8M",
            "threshold": 0.5,
            "chunk_size": 1024,
            "min_sentences_per_chunk": 5,
        }

    def test_kwarg_resolved_to_min_sentences_per_chunk_on_new_chonkie(self):
        # Regression for the silent-kwarg-drop bug: chonkie ≥1.5 renamed
        # this from `min_sentences` to `min_sentences_per_chunk`. The
        # strategy inspects the chonkie signature and forwards whichever
        # name the installed chonkie accepts.
        cls = _make_fake_chunker_cls(
            kwarg_name="min_sentences_per_chunk",
            tokenizer_is_property=True,
            chunks=[],
        )
        with patch.dict("sys.modules", {"chonkie": MagicMock(SemanticChunker=cls)}):
            import impresso_text_embedder.chunking.semantic as semantic_mod
            strat = semantic_mod.SemanticStrategy(min_sentences_per_chunk=7)
        assert strat._chunker._init_kwargs["min_sentences_per_chunk"] == 7
        assert "min_sentences" not in strat._chunker._init_kwargs

    def test_kwarg_resolved_to_min_sentences_on_old_chonkie(self):
        # On chonkie <1.5 the kwarg was `min_sentences`. The strategy
        # forwards under that name and the documented value still lands.
        cls = _make_fake_chunker_cls(
            kwarg_name="min_sentences",
            tokenizer_is_property=False,
            chunks=[],
        )
        with patch.dict("sys.modules", {"chonkie": MagicMock(SemanticChunker=cls)}):
            import impresso_text_embedder.chunking.semantic as semantic_mod
            strat = semantic_mod.SemanticStrategy(min_sentences_per_chunk=7)
        assert strat._chunker._init_kwargs["min_sentences"] == 7
        assert "min_sentences_per_chunk" not in strat._chunker._init_kwargs

    def test_tokenizer_override_swaps_via_underscore_attr_on_new_chonkie(self):
        # chonkie 1.5+ exposes `tokenizer` as a read-only property. The
        # strategy must reach through to `_tokenizer`.
        cls = _make_fake_chunker_cls(
            kwarg_name="min_sentences_per_chunk",
            tokenizer_is_property=True,
            chunks=[],
        )
        my_tok = MagicMock()
        my_tok.encode.return_value = [1, 2, 3]
        with patch.dict("sys.modules", {"chonkie": MagicMock(SemanticChunker=cls)}):
            import impresso_text_embedder.chunking.semantic as semantic_mod
            strat = semantic_mod.SemanticStrategy(tokenizer=my_tok)
        adapter = strat._chunker._tokenizer
        assert adapter is not None
        # adapter forwards to the wrapped tokenizer
        assert adapter.count_tokens("hi") == 3
        my_tok.encode.assert_called_with("hi", add_special_tokens=False)

    def test_tokenizer_override_swaps_via_public_attr_on_old_chonkie(self):
        # chonkie <1.5 has `tokenizer` as a plain settable attribute; the
        # strategy must take that path, not crash trying to set it.
        cls = _make_fake_chunker_cls(
            kwarg_name="min_sentences",
            tokenizer_is_property=False,
            chunks=[],
        )
        my_tok = MagicMock()
        with patch.dict("sys.modules", {"chonkie": MagicMock(SemanticChunker=cls)}):
            import impresso_text_embedder.chunking.semantic as semantic_mod
            strat = semantic_mod.SemanticStrategy(tokenizer=my_tok)
        assert strat._chunker.tokenizer is not None
        assert hasattr(strat._chunker.tokenizer, "count_tokens_batch")

    def test_tokenizer_omitted_leaves_chunker_tokenizer_alone(self):
        # When no tokenizer kwarg is passed, chonkie's default tokenizer
        # stays in place — out-of-sweep callers keep the historical
        # WordPiece-sized chunk_size behaviour.
        _, chunker, _ = self._make([])
        # _tokenizer was set to None in our stub; the strategy didn't
        # overwrite it with an adapter.
        assert chunker._tokenizer is None

    def test_adapter_count_tokens_batch_uses_input_ids(self):
        # The adapter is the only contact surface between our code and
        # chonkie's chunk-sizing loop. Verify it returns realistic
        # per-text counts when chonkie batches.
        from impresso_text_embedder.chunking.semantic import _TokenizerAdapter

        fake_tok = MagicMock()
        fake_tok.return_value = {"input_ids": [[1, 2, 3], [4, 5], []]}
        adapter = _TokenizerAdapter(fake_tok)
        assert adapter.count_tokens_batch(["a", "b", "c"]) == [3, 2, 0]
        # Empty input is short-circuited.
        assert adapter.count_tokens_batch([]) == []

    def test_empty_text_short_circuits_without_calling_chunker(self):
        strat, chunker, _ = self._make([])
        assert strat.chunk("") == []
        assert not hasattr(chunker, "_chunked_text")

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
        strat, chunker, _ = self._make(None)
        assert strat.chunk("something") == []
        assert chunker._chunked_text == "something"

    def test_get_strategy_forwards_kwargs_to_semantic_chunker(self):
        cls = _make_fake_chunker_cls(
            kwarg_name="min_sentences_per_chunk",
            tokenizer_is_property=True,
            chunks=[],
        )
        with patch.dict("sys.modules", {"chonkie": MagicMock(SemanticChunker=cls)}):
            strat = chunking.get_strategy("semantic", chunk_size=512)
        assert strat._chunker._init_kwargs["chunk_size"] == 512

    def test_no_kwarg_match_raises(self):
        # Defensive: chonkie removes both kwargs in some future release.
        # The strategy raises with a clear, actionable message instead
        # of silently dropping the value.
        class _FakeNoSentencesKwarg:
            def __init__(self, embedding_model, threshold, chunk_size, **kw):
                pass

            def chunk(self, text):
                return []

        with patch.dict(
            "sys.modules", {"chonkie": MagicMock(SemanticChunker=_FakeNoSentencesKwarg)}
        ):
            import impresso_text_embedder.chunking.semantic as semantic_mod
            with pytest.raises(RuntimeError, match=r"min_sentences"):
                semantic_mod.SemanticStrategy()
