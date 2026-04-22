"""Semantic chunking via ``chonkie.SemanticChunker``.

Configuration is ported from ``main:lib/text_embedding_processor.py`` so outputs
stay bit-compatible with the pre-migration pipeline. The underlying sentence
similarity model is ``minishlab/potion-base-8M`` — separate from the main
Alibaba-NLP embedder used downstream by :mod:`impresso_text_embedder.model`.
"""

from __future__ import annotations

from impresso_text_embedder.chunking.base import Chunk, ChunkingStrategy

DEFAULT_EMBEDDING_MODEL = "minishlab/potion-base-8M"
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_MIN_SENTENCES = 5


class SemanticStrategy(ChunkingStrategy):
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        min_sentences: int = DEFAULT_MIN_SENTENCES,
    ) -> None:
        from chonkie import SemanticChunker

        self._chunker = SemanticChunker(
            embedding_model=embedding_model,
            threshold=threshold,
            chunk_size=chunk_size,
            min_sentences=min_sentences,
        )

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        raw_chunks = self._chunker.chunk(text) or []
        return [
            Chunk(text=c.text, start=_extract_start(c))
            for c in raw_chunks
        ]


def _extract_start(chonkie_chunk: object) -> int | None:
    """Pull the character offset from a chonkie chunk if the attribute exists."""
    for attr in ("start", "start_char"):
        val = getattr(chonkie_chunk, attr, None)
        if val is not None:
            return int(val)
    return None
