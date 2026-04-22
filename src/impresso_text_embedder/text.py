"""Text reconstruction helpers.

Ported verbatim from ``main:lib/text_embedding_processor.py`` so the behaviour —
including how gaps between tokens are padded with spaces — matches the historical
outputs bit-for-bit.
"""

from __future__ import annotations

from typing import Any


def rebuild_ft_from_offsets(sents: list[dict[str, Any]]) -> str:
    """Reconstruct a document's full text from token offsets.

    Tokens are flattened across sentences, sorted by their character offset ``o``,
    and gaps are filled with spaces. Leading padding is preserved (text can start
    past position 0). Matches the legacy helper in the ``main`` branch.
    """
    toks: list[dict[str, Any]] = []
    for sent in sents:
        toks.extend(sent.get("tok", []))

    if not toks:
        return ""

    toks = sorted(toks, key=lambda x: x.get("o", 0))

    text: list[str] = []
    current_pos = 0

    for tok in toks:
        token_text = tok["t"]
        offset = tok["o"]

        if offset > current_pos:
            text.append(" " * (offset - current_pos))

        text.append(token_text)
        current_pos = offset + len(token_text)

    return "".join(text)


def rebuild_sentence_from_offsets(sent: dict[str, Any]) -> str:
    """Reconstruct a single sentence's text from its tokens.

    Unlike :func:`rebuild_ft_from_offsets`, the starting position is the first
    token's offset (no leading padding), and the result is stripped.
    """
    toks = sorted(sent.get("tok", []), key=lambda x: x["o"])

    if not toks:
        return ""

    text: list[str] = []
    current_pos = toks[0]["o"]

    for tok in toks:
        offset = tok.get("o", current_pos)
        token_text = tok.get("t", "")

        if offset > current_pos:
            text.append(" " * (offset - current_pos))

        text.append(token_text)
        current_pos = offset + len(token_text)

    return "".join(text).strip()
