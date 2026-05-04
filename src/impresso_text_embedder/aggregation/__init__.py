"""Aggregation strategies for combining chunk embeddings into one vector.

See ``.history/long-doc-chunking/notes.md`` for the full design-space.
``mean``, ``max``, ``first-chunk``, and ``length-weighted`` are
registered today; more strategies plug in via :func:`register_strategy`
without touching callers.
"""

from impresso_text_embedder.aggregation.base import (
    AggregationStrategy,
    available_strategies,
    get_strategy,
    register_strategy,
)
from impresso_text_embedder.aggregation.first_chunk import FirstChunkStrategy
from impresso_text_embedder.aggregation.length_weighted import (
    LengthWeightedMeanStrategy,
)
from impresso_text_embedder.aggregation.max_pool import MaxPoolStrategy
from impresso_text_embedder.aggregation.mean import MeanPoolStrategy

register_strategy("mean", MeanPoolStrategy)
register_strategy("max", MaxPoolStrategy)
register_strategy("first-chunk", FirstChunkStrategy)
register_strategy("length-weighted", LengthWeightedMeanStrategy)


__all__ = [
    "AggregationStrategy",
    "FirstChunkStrategy",
    "LengthWeightedMeanStrategy",
    "MaxPoolStrategy",
    "MeanPoolStrategy",
    "available_strategies",
    "get_strategy",
    "register_strategy",
]
