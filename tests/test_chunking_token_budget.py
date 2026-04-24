from __future__ import annotations

import pytest

from impresso_text_embedder import chunking
from impresso_text_embedder.chunking.base import Chunk
from impresso_text_embedder.chunking.token_budget import TokenBudgetStrategy


def _word_counter(text: str) -> int:
    """Trivial counter for testing: 1 token per whitespace-split word."""
    return len(text.split())


class TestTokenBudgetStrategy:
    def test_empty_text_returns_empty(self):
        strat = TokenBudgetStrategy(token_counter=_word_counter, max_tokens=10)
        assert strat.chunk("") == []

    def test_single_short_sentence_one_chunk(self):
        strat = TokenBudgetStrategy(token_counter=_word_counter, max_tokens=10)
        out = strat.chunk("Hello world.")
        assert len(out) == 1
        assert out[0].text == "Hello world."

    def test_sentences_packed_into_budget(self):
        # Each sentence is 2 tokens ("word. "); budget 4 tokens per chunk
        # Expect pairs of sentences grouped together.
        text = "foo bar. baz qux. alpha beta. gamma delta."
        strat = TokenBudgetStrategy(token_counter=_word_counter, max_tokens=4)
        out = strat.chunk(text)
        # 4 sentences × 2 tokens = 8 tokens total → 2 chunks of 4 each
        assert len(out) == 2
        assert out[0].text == "foo bar. baz qux."
        assert out[1].text == "alpha beta. gamma delta."

    def test_single_sentence_over_budget_emitted_as_own_chunk(self):
        # One sentence longer than the budget — still emitted (caller's problem
        # to warn / tokenizer truncates), but flushed separately.
        strat = TokenBudgetStrategy(token_counter=_word_counter, max_tokens=3)
        text = "Short one. This is a much longer sentence here."
        out = strat.chunk(text)
        # "Short one." (2 tok) fits; "This is a much longer sentence here." (7 tok) over-budget
        assert len(out) == 2
        assert out[0].text == "Short one."
        assert out[1].text.startswith("This is a much longer")

    def test_no_sentence_break_returns_whole_text(self):
        # No `.!?` boundary — the default regex returns the whole string as
        # one sentence, and we emit it as a single chunk regardless of size.
        strat = TokenBudgetStrategy(token_counter=_word_counter, max_tokens=3)
        text = "one two three four five six"  # no terminator at all
        out = strat.chunk(text)
        assert len(out) == 1
        assert out[0].text == text

    def test_start_offset_is_none_by_design(self):
        # We do not compute character offsets for long-doc chunking (they
        # aren't surfaced in the aggregated text-level output). Start is
        # always None.
        strat = TokenBudgetStrategy(token_counter=_word_counter, max_tokens=100)
        [c] = strat.chunk("Hello world. Another sentence.")
        assert c.start is None

    def test_custom_sentence_splitter(self):
        strat = TokenBudgetStrategy(
            token_counter=_word_counter,
            max_tokens=3,
            sentence_splitter=lambda t: t.split("|"),
        )
        out = strat.chunk("a b|c d|e f")
        # Each split piece is 2 tokens; budget 3 → one piece per chunk (the
        # next piece would overflow from 2→4)
        assert [c.text for c in out] == ["a b", "c d", "e f"]

    def test_rejects_non_positive_max_tokens(self):
        with pytest.raises(ValueError):
            TokenBudgetStrategy(token_counter=_word_counter, max_tokens=0)

    def test_registry_entry(self):
        # The package __init__ registers 'token-budget' with a kwargs-capable
        # factory. Must be callable with our ctor signature.
        strat = chunking.get_strategy(
            "token-budget", token_counter=_word_counter, max_tokens=5
        )
        assert isinstance(strat, TokenBudgetStrategy)
        out = strat.chunk("one two three. four five six.")
        assert len(out) >= 1
        assert all(isinstance(c, Chunk) for c in out)
