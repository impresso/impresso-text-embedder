"""Chunking base types and strategy registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A chunk of source text, optionally positioned by starting character offset."""

    text: str
    start: int | None = None


class ChunkingStrategy(ABC):
    """Interface for chunking strategies. Implementations are stateless after ``__init__``."""

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into an ordered list of :class:`Chunk` instances."""


_FACTORIES: dict[str, Callable[[], ChunkingStrategy]] = {}


def register_strategy(name: str, factory: Callable[[], ChunkingStrategy]) -> None:
    """Register ``factory`` under ``name``. Overwrites any previous registration."""
    if not name:
        raise ValueError("strategy name must be non-empty")
    _FACTORIES[name] = factory


def get_strategy(name: str) -> ChunkingStrategy:
    """Build and return a fresh strategy instance for ``name``."""
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown chunking strategy {name!r}; known: {available_strategies()}"
        ) from exc
    return factory()


def available_strategies() -> list[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(_FACTORIES)
