"""Aggregation strategies for combining chunk embeddings into one vector.

See ``.progress/long-doc-chunking/notes.md`` for the full design-space.
Only ``mean`` is registered today; more strategies plug in via
:func:`register_strategy` without touching callers.
"""

from impresso_text_embedder.aggregation.base import (
    AggregationStrategy,
    available_strategies,
    get_strategy,
    register_strategy,
)
from impresso_text_embedder.aggregation.mean import MeanPoolStrategy

register_strategy("mean", MeanPoolStrategy)


__all__ = [
    "AggregationStrategy",
    "MeanPoolStrategy",
    "available_strategies",
    "get_strategy",
    "register_strategy",
]
