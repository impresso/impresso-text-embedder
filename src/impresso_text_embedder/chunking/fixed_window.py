"""Fixed-size token window chunking — no overlap, no sentence awareness.

The dumbest chunker that works: tokenize the full text once with the model's
own tokenizer, slice the flat token-id list into contiguous non-overlapping
windows of ``max_tokens`` ids, decode each slice back to a string. The
resulting :class:`Chunk` list is fed to ``model.encode`` downstream, which
re-tokenises each decoded string — a known redundancy (see the "Optimization
level — L1" section of ``.history/long-doc-chunking/notes.md``) kept
deliberately for simplicity. L2/L3 optimisations are deferred.

Chosen as the step-16 default over :class:`TokenBudgetStrategy` because:

* no sentence-splitter regex to tune or fail on OCR'd text,
* no dependency surface beyond the model's tokenizer,
* trivially correct — bugs have nowhere to hide in ~20 LOC,
* `token-budget` remains registered for the day its sentence-aware
  packing proves better on multi-topic long docs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from impresso_text_embedder.chunking.base import Chunk, ChunkingStrategy

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class FixedWindowStrategy(ChunkingStrategy):
    """Contiguous fixed-size token-window chunker.

    Parameters
    ----------
    tokenizer
        Any HF-style tokenizer exposing ``encode(text, add_special_tokens=…)``
        and ``decode(ids, skip_special_tokens=…)``. In production this is the
        loaded ``SentenceTransformer``'s ``.tokenizer``; tests pass a stub
        with the same two methods.
    max_tokens
        Window size in subword tokens. Typically
        ``tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=False)``
        so the encoder's own CLS/SEP addition brings each chunk up to exactly
        its configured max (e.g. 8190 for gte-multilingual-base so the encoder
        sees 8192 after CLS/SEP).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self._tokenizer = tokenizer
        self._max = max_tokens

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []
        chunks: list[Chunk] = []
        for start in range(0, len(token_ids), self._max):
            window = token_ids[start : start + self._max]
            piece = self._tokenizer.decode(window, skip_special_tokens=True)
            chunks.append(Chunk(text=piece, start=None))
        return chunks
