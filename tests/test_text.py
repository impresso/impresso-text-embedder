from impresso_text_embedder.text import (
    rebuild_ft_from_offsets,
    rebuild_sentence_from_offsets,
)


def _tok(t: str, o: int) -> dict:
    return {"t": t, "o": o}


class TestRebuildFtFromOffsets:
    def test_single_sentence_contiguous(self):
        sents = [{"tok": [_tok("Hello", 0), _tok("world", 6)]}]
        assert rebuild_ft_from_offsets(sents) == "Hello world"

    def test_multiple_sentences_flattened_and_sorted(self):
        sents = [
            {"tok": [_tok("two", 12)]},
            {"tok": [_tok("One.", 0), _tok("is", 5), _tok("here.", 8)]},
        ]
        assert rebuild_ft_from_offsets(sents) == "One. is here.two"

    def test_leading_gap_preserved(self):
        sents = [{"tok": [_tok("late", 4)]}]
        assert rebuild_ft_from_offsets(sents) == "    late"

    def test_empty_input(self):
        assert rebuild_ft_from_offsets([]) == ""
        assert rebuild_ft_from_offsets([{"tok": []}]) == ""


class TestRebuildSentenceFromOffsets:
    def test_basic(self):
        sent = {"tok": [_tok("Hello", 0), _tok("world", 6)]}
        assert rebuild_sentence_from_offsets(sent) == "Hello world"

    def test_strips_internal_start_offset(self):
        # Even though first token starts at offset 10, result is stripped.
        sent = {"tok": [_tok("foo", 10), _tok("bar", 14)]}
        assert rebuild_sentence_from_offsets(sent) == "foo bar"

    def test_gaps_inside_sentence_padded(self):
        sent = {"tok": [_tok("a", 0), _tok("b", 5)]}
        assert rebuild_sentence_from_offsets(sent) == "a    b"

    def test_empty_tokens(self):
        assert rebuild_sentence_from_offsets({"tok": []}) == ""
        assert rebuild_sentence_from_offsets({}) == ""
