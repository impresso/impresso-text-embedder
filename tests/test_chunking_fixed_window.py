from __future__ import annotations

import pytest

from impresso_text_embedder import chunking
from impresso_text_embedder.chunking.base import Chunk
from impresso_text_embedder.chunking.fixed_window import FixedWindowStrategy


class _StubTokenizer:
    """Word-level stub that matches HF's `encode`/`decode` signatures.

    Tokenisation is ``text.split()`` with each word mapped to a unique int;
    decoding reverses the mapping and joins with single spaces. Close enough
    for slicing tests — no SentencePiece round-trip subtleties, which
    sidesteps the `Ġ`-prefix / whitespace-normalisation issues that would
    otherwise complicate the assertions.
    """

    def __init__(self, model_max_length: int = 100, n_special: int = 2) -> None:
        self.model_max_length = model_max_length
        self._n_special = n_special
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _intern(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._inv[idx] = word
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # add_special_tokens is accepted for API parity; this stub never
        # inserts specials, so the kwarg is ignored.
        return [self._intern(w) for w in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._inv[i] for i in ids)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._n_special


class TestFixedWindowStrategy:
    def test_empty_text_returns_empty(self):
        strat = FixedWindowStrategy(tokenizer=_StubTokenizer(), max_tokens=4)
        assert strat.chunk("") == []

    def test_single_short_doc_one_chunk(self):
        strat = FixedWindowStrategy(tokenizer=_StubTokenizer(), max_tokens=10)
        out = strat.chunk("hello world")
        assert len(out) == 1
        assert out[0].text == "hello world"
        assert out[0].start is None

    def test_exact_multiple_yields_exact_chunk_count(self):
        tok = _StubTokenizer()
        strat = FixedWindowStrategy(tokenizer=tok, max_tokens=3)
        text = "a b c d e f"  # 6 tokens, window 3 → exactly 2 chunks
        out = strat.chunk(text)
        assert [c.text for c in out] == ["a b c", "d e f"]

    def test_non_multiple_yields_short_final_chunk(self):
        tok = _StubTokenizer()
        strat = FixedWindowStrategy(tokenizer=tok, max_tokens=3)
        text = "a b c d e f g"  # 7 tokens → 3 chunks of sizes [3,3,1]
        out = strat.chunk(text)
        assert [c.text for c in out] == ["a b c", "d e f", "g"]

    def test_large_window_means_one_chunk(self):
        strat = FixedWindowStrategy(tokenizer=_StubTokenizer(), max_tokens=1000)
        text = "token " * 50
        out = strat.chunk(text.strip())
        assert len(out) == 1

    def test_rejects_non_positive_max_tokens(self):
        with pytest.raises(ValueError):
            FixedWindowStrategy(tokenizer=_StubTokenizer(), max_tokens=0)
        with pytest.raises(ValueError):
            FixedWindowStrategy(tokenizer=_StubTokenizer(), max_tokens=-5)

    def test_registry_entry(self):
        # fixed-window is registered at package import with a
        # kwargs-capable factory. Must be callable with the ctor signature.
        strat = chunking.get_strategy(
            "fixed-window", tokenizer=_StubTokenizer(), max_tokens=4
        )
        assert isinstance(strat, FixedWindowStrategy)
        out = strat.chunk("one two three four five")
        assert len(out) == 2
        assert all(isinstance(c, Chunk) for c in out)

    def test_round_trip_ids_preserved(self):
        """Decoding a window then re-encoding gives back the same ids.

        This is the retrieval-quality guarantee: the decoded chunk string,
        when passed to ``model.encode``, tokenises to the same ids the
        chunker sliced. For this stub the invariant is exact; for real
        SentencePiece tokenizers it holds modulo leading-whitespace
        normalisation (acceptable for CLS-pooled encoding).
        """
        tok = _StubTokenizer()
        strat = FixedWindowStrategy(tokenizer=tok, max_tokens=3)
        text = "alpha beta gamma delta epsilon zeta"
        chunks = strat.chunk(text)
        reencoded = [tok.encode(c.text, add_special_tokens=False) for c in chunks]
        original_ids = tok.encode(text, add_special_tokens=False)
        flat = [i for chunk_ids in reencoded for i in chunk_ids]
        assert flat == original_ids
