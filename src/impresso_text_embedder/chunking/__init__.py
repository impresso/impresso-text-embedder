"""Chunking strategies for texts exceeding the model's max sequence length.

See ``.progress/chunking/notes.md`` for the registry contract. The `semantic`
strategy is registered at package import time via a lazy factory so importing
this package does not pull in ``chonkie`` until the strategy is actually used.
"""

from impresso_text_embedder.chunking.base import (
    Chunk,
    ChunkingStrategy,
    available_strategies,
    get_strategy,
    register_strategy,
)


def _make_semantic() -> ChunkingStrategy:
    from impresso_text_embedder.chunking.semantic import SemanticStrategy

    return SemanticStrategy()


def _make_token_budget(**kwargs: object) -> ChunkingStrategy:
    from impresso_text_embedder.chunking.token_budget import TokenBudgetStrategy

    return TokenBudgetStrategy(**kwargs)  # type: ignore[arg-type]


def _make_fixed_window(**kwargs: object) -> ChunkingStrategy:
    from impresso_text_embedder.chunking.fixed_window import FixedWindowStrategy

    return FixedWindowStrategy(**kwargs)  # type: ignore[arg-type]


register_strategy("semantic", _make_semantic)
register_strategy("token-budget", _make_token_budget)
register_strategy("fixed-window", _make_fixed_window)


__all__ = [
    "Chunk",
    "ChunkingStrategy",
    "available_strategies",
    "get_strategy",
    "register_strategy",
]
