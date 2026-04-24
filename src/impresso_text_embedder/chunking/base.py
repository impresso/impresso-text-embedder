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


_FACTORIES: dict[str, Callable[..., ChunkingStrategy]] = {}


def register_strategy(
    name: str, factory: Callable[..., ChunkingStrategy]
) -> None:
    """Register ``factory`` under ``name``. Overwrites any previous registration.

    ``factory`` is any callable returning a :class:`ChunkingStrategy`. It may
    accept keyword arguments; callers pass them through :func:`get_strategy`.
    Zero-arg factories (e.g. the class itself when no runtime config is
    needed) are supported as the common case.
    """
    if not name:
        raise ValueError("strategy name must be non-empty")
    _FACTORIES[name] = factory


def get_strategy(name: str, **kwargs: object) -> ChunkingStrategy:
    """Build and return a fresh strategy instance for ``name``.

    Keyword arguments are forwarded to the registered factory so strategies
    that need runtime config (e.g. a tokenizer-derived ``token_counter``)
    can receive it at call time without the registry growing a superset of
    kwargs for every strategy.
    """
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown chunking strategy {name!r}; known: {available_strategies()}"
        ) from exc
    return factory(**kwargs)


def available_strategies() -> list[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(_FACTORIES)
