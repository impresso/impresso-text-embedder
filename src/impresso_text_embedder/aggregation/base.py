"""Aggregation base types and strategy registry.

An aggregation strategy collapses ``K`` per-chunk embeddings (shape
``[K, D]``) into a single ``D``-dim document vector. Used at
``--embedding-level text`` when a document was chunked because it
exceeded the model's max context (see
``.history/long-doc-chunking/notes.md``).

The registry mirrors :mod:`impresso_text_embedder.chunking.base` so new
strategies (length-weighted mean, max, attention-weighted, …) can slot
in without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class AggregationStrategy(ABC):
    """Interface for chunk-embedding aggregation strategies.

    Implementations must be stateless after ``__init__`` so a single
    instance is safe to share across documents.
    """

    @abstractmethod
    def aggregate(
        self,
        vectors: np.ndarray,
        weights: list[int] | None = None,
    ) -> np.ndarray:
        """Collapse a ``[K, D]`` array into a ``[D]`` unit vector.

        ``weights`` is an optional per-chunk weight (typically chunk
        token count). Strategies that don't use it should ignore it;
        strategies that do (e.g. length-weighted mean) should document
        the expected semantics. ``K`` is guaranteed ≥ 1 by the caller.
        """


_FACTORIES: dict[str, Callable[..., AggregationStrategy]] = {}


def register_strategy(
    name: str, factory: Callable[..., AggregationStrategy]
) -> None:
    """Register ``factory`` under ``name``. Overwrites any previous registration.

    ``factory`` is any callable that returns an :class:`AggregationStrategy`
    instance. Accepting keyword arguments is allowed so strategies needing
    runtime config (e.g. a decay coefficient) can receive it at
    :func:`get_strategy` call time; strategies with no config should
    register the class itself as the factory.
    """
    if not name:
        raise ValueError("strategy name must be non-empty")
    _FACTORIES[name] = factory


def get_strategy(name: str, **kwargs: object) -> AggregationStrategy:
    """Build and return a fresh aggregation strategy for ``name``.

    Keyword arguments are forwarded to the registered factory, which
    lets future strategies carry per-run config without hard-coding a
    superset of kwargs here.
    """
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown aggregation strategy {name!r}; known: {available_strategies()}"
        ) from exc
    return factory(**kwargs)


def available_strategies() -> list[str]:
    """Return the sorted list of registered aggregation strategy names."""
    return sorted(_FACTORIES)
