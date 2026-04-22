"""Per-record embedding logic for the three levels (text, sentence, chunk).

Keeps encoder calls GPU-friendly: text-level is batched across records via
:class:`TextBatcher`; sentence- and chunk-level encode many items per call
already, so they're done per record.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from impresso_text_embedder.chunking.base import ChunkingStrategy
from impresso_text_embedder.model import encode_texts
from impresso_text_embedder.schema import (
    ChunkItem,
    ChunkRecord,
    SentenceItem,
    SentenceRecord,
    TextRecord,
    utc_timestamp,
)
from impresso_text_embedder.text import (
    rebuild_ft_from_offsets,
    rebuild_sentence_from_offsets,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderConfig:
    """Runtime config threaded through the per-record / batcher calls."""

    batch_size: int
    normalize_embeddings: bool = True
    min_char_length: int = 400
    include_text: bool = False
    content_types: frozenset[str] = frozenset({"ar"})


def build_embedder_tag(model_name: str, model_revision: str | None) -> str:
    """Formatting convention matching the legacy `embedder` field (e.g. `name@revision`)."""
    return f"{model_name}@{model_revision or 'default'}"


def _passes_content_type(record: dict, cfg: EncoderConfig) -> bool:
    tp = record.get("tp")
    return tp is None or tp in cfg.content_types


# ---------------------------------------------------------------------------
# Text level — batched across records
# ---------------------------------------------------------------------------


@dataclass
class _PendingText:
    ci_id: str
    text: str


class TextBatcher:
    """Accumulates records into a buffer; flushes produce :class:`TextRecord` objects."""

    def __init__(
        self,
        model: SentenceTransformer,
        cfg: EncoderConfig,
        embedder_tag: str,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._embedder_tag = embedder_tag
        self._pending: list[_PendingText] = []

    def __len__(self) -> int:
        return len(self._pending)

    def add(self, record: dict) -> list[TextRecord]:
        """Add one record. Returns emitted records if a flush happened, else []."""
        if not _passes_content_type(record, self._cfg):
            return []
        ci_id = record.get("id")
        if not ci_id:
            log.debug("skipping record without id")
            return []
        text = rebuild_ft_from_offsets(record.get("sents", []))
        if len(text) <= self._cfg.min_char_length:
            return []
        self._pending.append(_PendingText(ci_id=ci_id, text=text))
        if len(self._pending) >= self._cfg.batch_size:
            return self.flush()
        return []

    def flush(self) -> list[TextRecord]:
        if not self._pending:
            return []
        texts = [p.text for p in self._pending]
        vecs = encode_texts(
            self._model,
            texts,
            batch_size=self._cfg.batch_size,
            normalize=self._cfg.normalize_embeddings,
        )
        ts = utc_timestamp()
        out: list[TextRecord] = []
        for p, vec in zip(self._pending, vecs, strict=True):
            out.append(
                TextRecord(
                    id=p.ci_id,
                    ts=ts,
                    embedder=self._embedder_tag,
                    len=len(p.text),
                    embedding=vec.tolist(),
                    text=p.text if self._cfg.include_text else None,
                )
            )
        self._pending.clear()
        return out


# ---------------------------------------------------------------------------
# Sentence level — one record at a time, many sentences per call
# ---------------------------------------------------------------------------


def embed_sentence_record(
    record: dict,
    model: SentenceTransformer,
    cfg: EncoderConfig,
) -> SentenceRecord | None:
    if not _passes_content_type(record, cfg):
        return None

    ci_id = record.get("id")
    if not ci_id:
        return None

    sents = record.get("sents") or []
    if not sents:
        log.debug("sentence level requested but no sents for %s", ci_id)
        return None

    kept: list[tuple[int, str, int | None]] = []
    for idx, s in enumerate(sents):
        txt = rebuild_sentence_from_offsets(s).strip()
        if len(txt) <= cfg.min_char_length:
            continue
        offset = s.get("o") if isinstance(s, dict) else None
        kept.append((idx, txt, offset))
    if not kept:
        return None

    texts = [t for _, t, _ in kept]
    vecs = encode_texts(
        model,
        texts,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize_embeddings,
    )

    lg = record.get("lg")
    items = [
        SentenceItem(
            sent_id=sid,
            embedding=vec.tolist(),
            size=len(vec),
            lg=lg,
            o=offset,
        )
        for (sid, _t, offset), vec in zip(kept, vecs, strict=True)
    ]
    return SentenceRecord(
        ts=utc_timestamp(),
        ci_id=ci_id,
        sents=items,
        lingproc_path=record.get("lingproc_path"),
    )


# ---------------------------------------------------------------------------
# Chunk level — one record at a time, many chunks per call
# ---------------------------------------------------------------------------


def embed_chunk_record(
    record: dict,
    model: SentenceTransformer,
    cfg: EncoderConfig,
    chunker: ChunkingStrategy,
) -> ChunkRecord | None:
    if not _passes_content_type(record, cfg):
        return None

    ci_id = record.get("id")
    if not ci_id:
        return None

    text = rebuild_ft_from_offsets(record.get("sents") or [])
    if len(text) <= cfg.min_char_length:
        return None

    chunks = chunker.chunk(text)
    if not chunks:
        return None

    texts = [c.text for c in chunks]
    vecs = encode_texts(
        model,
        texts,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize_embeddings,
    )

    lg = record.get("lg")
    items = [
        ChunkItem(
            chunk_id=idx,
            embedding=vec.tolist(),
            size=len(vec),
            lg=lg,
            o=c.start,
        )
        for idx, (c, vec) in enumerate(zip(chunks, vecs, strict=True))
    ]
    return ChunkRecord(
        ts=utc_timestamp(),
        ci_id=ci_id,
        chunks=items,
        lingproc_path=record.get("lingproc_path"),
    )


# ---------------------------------------------------------------------------
# Convenience: iterate an input and yield output dicts by level
# ---------------------------------------------------------------------------


def embed_records(
    records: Iterable[dict],
    level: str,
    model: SentenceTransformer,
    cfg: EncoderConfig,
    embedder_tag: str,
    chunker: ChunkingStrategy | None = None,
) -> Iterable[dict]:
    """Yield output dicts (already `.to_dict()`) for all input records.

    Callers pick ``level`` and pass the corresponding helpers. ``chunker`` is
    required when ``level == "chunk"``.
    """
    if level == "text":
        batcher = TextBatcher(model, cfg, embedder_tag)
        for rec in records:
            for out in batcher.add(rec):
                yield out.to_dict()
        for out in batcher.flush():
            yield out.to_dict()
    elif level == "sentence":
        for rec in records:
            out = embed_sentence_record(rec, model, cfg)
            if out is not None:
                yield out.to_dict()
    elif level == "chunk":
        if chunker is None:
            raise ValueError("chunking strategy required for level='chunk'")
        for rec in records:
            out = embed_chunk_record(rec, model, cfg, chunker)
            if out is not None:
                yield out.to_dict()
    else:
        raise ValueError(f"unknown embedding level: {level!r}")
