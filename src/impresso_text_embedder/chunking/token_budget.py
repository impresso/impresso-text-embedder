"""Sentence-aware token-budget packing.

Greedy: iterate sentences, pack them into ``max_tokens`` buckets using the
caller-supplied ``token_counter``. A sentence larger than the budget is
emitted as its own (over-budget) chunk; the encoder will truncate it.
Preserves sentence boundaries — never splits mid-sentence.

Rationale for picking this strategy as the first token-aware chunker:
``.progress/long-doc-chunking/notes.md`` §"Question 1 — Chunking strategies"
option D.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from impresso_text_embedder.chunking.base import Chunk, ChunkingStrategy

# Default ceiling: slightly below the model's 8192-token max to leave room
# for CLS/SEP without having to know them here. Calibration on Impresso
# data is OPEN (see notes OPEN-O2).
DEFAULT_MAX_TOKENS = 8000

# Sentence split on any `.!?` followed by whitespace. Intentionally
# simple — OCR'd Impresso text has noisy casing/punctuation and records
# usually carry a pre-computed ``sents`` field we can use when present
# (wiring deferred). Over-splitting on constructs like "Mr. Smith" or
# mid-sentence abbreviations is acceptable: the packer downstream only
# needs correct boundaries to flush on, and smaller-than-budget chunks
# just means more chunks per doc — not a correctness issue.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _default_sentence_split(text: str) -> list[str]:
    """Split on common sentence terminators. Empty fragments are dropped."""
    return [s for s in _SENT_RE.split(text) if s and not s.isspace()]


class TokenBudgetStrategy(ChunkingStrategy):
    """Sentence-aware greedy packer targeting a maximum token budget per chunk.

    Parameters
    ----------
    token_counter
        Callable ``text -> int`` returning a token count. In production
        this wraps the model's tokenizer; in tests, pass a fake for
        speed. Special tokens (CLS/SEP) are not the caller's concern —
        budget the caller picks should leave slack for them.
    max_tokens
        Upper bound on tokens per emitted chunk. Defaults to
        :data:`DEFAULT_MAX_TOKENS`.
    sentence_splitter
        Optional custom splitter. Defaults to a simple regex split.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sentence_splitter: Callable[[str], list[str]] | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self._count = token_counter
        self._max = max_tokens
        self._split = sentence_splitter or _default_sentence_split

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        sents = self._split(text)
        if not sents:
            # No sentence break detected — return the whole text as a
            # single chunk. Over-budget inputs here will be truncated
            # by the encoder; logging happens upstream.
            return [Chunk(text=text, start=None)]

        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0

        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            s_tokens = self._count(sent)

            # Sentence alone exceeds budget: flush current buffer, then
            # emit the oversized sentence as its own chunk (encoder
            # will truncate it — this is the documented escape hatch).
            if s_tokens > self._max:
                if buf:
                    chunks.append(Chunk(text=" ".join(buf), start=None))
                    buf, buf_tokens = [], 0
                chunks.append(Chunk(text=sent, start=None))
                continue

            # Adding this sentence would overflow: flush first.
            if buf and buf_tokens + s_tokens > self._max:
                chunks.append(Chunk(text=" ".join(buf), start=None))
                buf, buf_tokens = [], 0

            buf.append(sent)
            buf_tokens += s_tokens

        if buf:
            chunks.append(Chunk(text=" ".join(buf), start=None))
        return chunks
