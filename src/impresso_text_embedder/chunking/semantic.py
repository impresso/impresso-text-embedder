"""Semantic chunking via ``chonkie.SemanticChunker``.

Defaults are ported from ``main:lib/text_embedding_processor.py`` so outputs
stay close to the pre-migration pipeline. The underlying sentence
similarity model is ``minishlab/potion-base-8M`` — separate from the main
Alibaba-NLP embedder used downstream by :mod:`impresso_text_embedder.model`.

Two non-obvious wiring details, each tied to a chonkie-internals
workaround. Both deliberately avoid pinning chonkie to a specific
version — chonkie 1.5+ pins ``numpy>=2.0``, which is incompatible with
the production container's load-bearing ``numpy==1.26.4`` (NGC ABI
stack). We therefore probe chonkie's surface at construction time and
adapt; the same module works on chonkie 0.x/1.0–1.4 (NGC-compatible)
and on chonkie ≥1.5 (local dev / future numpy-2 stacks).

- **min-sentences kwarg** — chonkie ≥1.5 calls this
  ``min_sentences_per_chunk``; older versions called it
  ``min_sentences``. We inspect ``SemanticChunker.__init__``'s signature
  and forward whichever the installed chonkie accepts. Forwarding the
  wrong name lands the value in ``**kwargs`` and is silently dropped —
  the floor reverts to chonkie's default (``=1``), making chunks
  smaller than the contract says.
- ``tokenizer`` (optional) — overrides the unit in which ``chunk_size``
  is measured. By default ``SemanticChunker.__init__`` does
  ``self._tokenizer = AutoTokenizer(self.embedding_model.get_tokenizer())``,
  which on ``potion-base-8M`` is a 30k-vocab WordPiece tokenizer —
  substantially more aggressive than the GTE multilingual SentencePiece
  tokenizer (≈1.5–1.8× more tokens for the same fr/de text). For the
  chunking-eval sweep we must size chunks in the **encoder's** tokens
  so a "chunk_size=4096" target is comparable across the
  fixed-window / token-budget / semantic families. We swap the
  underlying chunker tokenizer with :class:`_TokenizerAdapter`, an
  in-house counter object that exposes the single method
  (``count_tokens_batch``) chonkie's ``SemanticChunker._split_sentences``
  calls — no dependency on ``chonkie.tokenizer.AutoTokenizer`` (which
  doesn't exist in chonkie <1.5). Touching ``_tokenizer`` /
  ``tokenizer`` is a deliberate private-API reach; if a future chonkie
  release adds a public setter or a ``tokenizer=`` constructor kwarg on
  ``SemanticChunker``, prefer the public API.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from impresso_text_embedder.chunking.base import Chunk, ChunkingStrategy

DEFAULT_EMBEDDING_MODEL = "minishlab/potion-base-8M"
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_MIN_SENTENCES_PER_CHUNK = 5


class _TokenizerAdapter:
    """Minimal chonkie-tokenizer adapter wrapping an HF transformers tokenizer.

    Chonkie's ``SemanticChunker`` calls a single method on its tokenizer
    (``count_tokens_batch``, see chonkie 1.6 ``chunker/semantic.py:201``);
    earlier versions also call ``count_tokens`` from ``BaseChunker``. We
    expose both, plus a no-op ``encode`` for chonkie versions that
    happen to call it. The adapter does not subclass any chonkie type so
    it works regardless of which chonkie module layout is installed.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        # HF fast tokenizers handle list-input batch encoding natively.
        out = self._tok(list(texts), add_special_tokens=False)
        return [len(ids) for ids in out["input_ids"]]

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))


class SemanticStrategy(ChunkingStrategy):
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        min_sentences_per_chunk: int = DEFAULT_MIN_SENTENCES_PER_CHUNK,
        tokenizer: Any | None = None,
    ) -> None:
        from chonkie import SemanticChunker

        ctor_kwargs: dict[str, Any] = {
            "embedding_model": embedding_model,
            "threshold": threshold,
            "chunk_size": chunk_size,
        }
        ctor_kwargs[_resolve_min_sentences_kwarg(SemanticChunker)] = (
            min_sentences_per_chunk
        )
        self._chunker = SemanticChunker(**ctor_kwargs)
        if tokenizer is not None:
            _swap_chunker_tokenizer(self._chunker, _TokenizerAdapter(tokenizer))

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        raw_chunks = self._chunker.chunk(text) or []
        return [
            Chunk(text=c.text, start=_extract_start(c))
            for c in raw_chunks
        ]


def _resolve_min_sentences_kwarg(chunker_cls: type) -> str:
    """Pick the right kwarg name for the installed chonkie."""
    params = inspect.signature(chunker_cls.__init__).parameters
    if "min_sentences_per_chunk" in params:
        return "min_sentences_per_chunk"
    if "min_sentences" in params:
        return "min_sentences"
    raise RuntimeError(
        "chonkie.SemanticChunker has neither 'min_sentences_per_chunk' nor "
        "'min_sentences' in its signature. Update SemanticStrategy."
    )


def _swap_chunker_tokenizer(chunker: object, adapter: _TokenizerAdapter) -> None:
    """Replace chonkie's internal tokenizer with our adapter.

    chonkie 1.6 stores it on ``_tokenizer`` and exposes a read-only
    ``tokenizer`` property; older versions store it on ``tokenizer``
    directly. We try the public name first (works on old chonkie),
    fall through to ``_tokenizer`` (chonkie 1.5+).
    """
    try:
        chunker.tokenizer = adapter  # type: ignore[attr-defined]
        return
    except AttributeError:
        pass
    # New chonkie's `tokenizer` is a read-only property; reach through.
    chunker._tokenizer = adapter  # type: ignore[attr-defined]


def _extract_start(chonkie_chunk: object) -> int | None:
    """Pull the character offset from a chonkie chunk if the attribute exists."""
    for attr in ("start", "start_char"):
        val = getattr(chonkie_chunk, attr, None)
        if val is not None:
            return int(val)
    return None
