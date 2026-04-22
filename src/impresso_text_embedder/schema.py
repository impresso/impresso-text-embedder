"""Output schemas for the three embedding levels (text, sentence, chunk).

Mirrors the shapes emitted by ``main:lib/text_embedding_processor.py``. Float values
in embeddings are rounded to 5 decimals on serialization (legacy behaviour). Optional
fields with ``None`` are omitted from the output so the JSONL lines don't carry
null keys.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any

EMBEDDING_DECIMALS = 5


def utc_timestamp() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_embedding(embedding: list[float]) -> list[float]:
    return [round(float(x), EMBEDDING_DECIMALS) for x in embedding]


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Text level (flat, one embedding per document)
# ---------------------------------------------------------------------------


@dataclass
class TextRecord:
    """One embedding per content item. Written as a flat JSON line."""

    id: str
    ts: str
    embedder: str
    len: int
    embedding: list[float]
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding"] = _round_embedding(self.embedding)
        return _drop_none(d)


# ---------------------------------------------------------------------------
# Sentence level
# ---------------------------------------------------------------------------


@dataclass
class SentenceItem:
    sent_id: int
    embedding: list[float]
    size: int
    lg: str | None = None
    o: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding"] = _round_embedding(self.embedding)
        return _drop_none(d)


@dataclass
class SentenceRecord:
    """One record per content item; ``sents`` lists per-sentence embeddings."""

    ts: str
    ci_id: str
    sents: list[SentenceItem] = field(default_factory=list)
    model_id: str | None = None
    lingproc_path: str | None = None
    git: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ts": self.ts,
            "ci_id": self.ci_id,
            "sents": [s.to_dict() for s in self.sents],
            "model_id": self.model_id,
            "lingproc_path": self.lingproc_path,
            "git": self.git,
        }
        return _drop_none(d)


# ---------------------------------------------------------------------------
# Chunk level
# ---------------------------------------------------------------------------


@dataclass
class ChunkItem:
    chunk_id: int
    embedding: list[float]
    size: int
    lg: str | None = None
    o: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding"] = _round_embedding(self.embedding)
        return _drop_none(d)


@dataclass
class ChunkRecord:
    """One record per content item; ``chunks`` lists per-chunk embeddings."""

    ts: str
    ci_id: str
    chunks: list[ChunkItem] = field(default_factory=list)
    model_id: str | None = None
    lingproc_path: str | None = None
    git: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ts": self.ts,
            "ci_id": self.ci_id,
            "chunks": [c.to_dict() for c in self.chunks],
            "model_id": self.model_id,
            "lingproc_path": self.lingproc_path,
            "git": self.git,
        }
        return _drop_none(d)
