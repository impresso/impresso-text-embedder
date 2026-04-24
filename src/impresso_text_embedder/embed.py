"""Per-record embedding logic for the three levels (text, sentence, chunk).

Keeps encoder calls GPU-friendly: text-level is batched across records via
:class:`TextBatcher`; sentence- and chunk-level encode many items per call
already, so they're done per record.

Long-document handling at text level (design:
``.progress/long-doc-chunking/notes.md``):

A document exceeding the model's max context is silently truncated by the
tokenizer by default. When :attr:`EncoderConfig.long_doc` is configured
with ``strategy="chunk"``, :class:`TextBatcher` instead chunks the
document, encodes the chunks alongside other pending texts in the same
batched ``model.encode`` call, and aggregates the resulting K vectors
into one document vector via the configured aggregation strategy. Only
**mean pool + L2 renormalise** is registered today; other strategies
plug in via the :mod:`impresso_text_embedder.aggregation` registry.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from impresso_text_embedder.aggregation.base import AggregationStrategy

log = logging.getLogger(__name__)

# Filter-reason identifiers. Used both as the ``reason=…`` field in per-record
# DEBUG logs and as keys in the per-file filter tally counter that feeds the
# "done" INFO line. Single source of truth so the log and the tally can't drift.
FILTER_CONTENT_TYPE = "content_type"
FILTER_MISSING_CONTENT_TYPE = "missing_content_type"
FILTER_MISSING_ID = "missing_id"
FILTER_TOO_SHORT = "too_short"
FILTER_NO_SENTENCES = "no_sentences"
FILTER_SENTENCE_TOO_SHORT = "sentence_too_short"
FILTER_NO_CHUNKS = "no_chunks"

# Long-doc telemetry tally keys. Not filter reasons (the record is still
# emitted), just counters surfaced in the per-file "done" line so the
# operator can see how often the long-doc branch fired.
LONG_DOC_CHUNKED = "long_doc_chunked"

# Token count above which a text is treated as "long" and routed to the
# chunk-and-aggregate path. See :class:`LongDocConfig` for the full
# detection pipeline (cheap char-estimate gate + optional real tokenise).
DEFAULT_CHARS_PER_TOKEN = 3.0


# ---------------------------------------------------------------------------
# Token counting protocol
# ---------------------------------------------------------------------------

# A ``TokenCounter`` is any callable ``text -> int``. In production this
# wraps the model's tokenizer (``lambda t: len(tokenizer.encode(t,
# add_special_tokens=False))``); in tests it can be a trivial stub like
# ``lambda t: len(t.split())``.
TokenCounter = Callable[[str], int]


# ---------------------------------------------------------------------------
# Long-doc config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LongDocConfig:
    """How to handle text-level documents longer than the model's context.

    ``strategy="truncate"`` preserves the pre-step-16 behaviour: let the
    tokenizer's default ``truncation=True`` drop the tail. ``"chunk"``
    enables the detect → chunk → encode → aggregate path.

    When ``strategy="chunk"``, both ``chunker`` and ``aggregator`` must
    be provided. ``token_counter`` is used to confirm that the cheap
    character-based estimate really did clear the ``model_max_tokens``
    ceiling; without it, the short path always wins (detection is a
    no-op), which effectively reduces to ``"truncate"``.
    """

    strategy: Literal["truncate", "chunk"] = "truncate"
    chunker: ChunkingStrategy | None = None
    aggregator: AggregationStrategy | None = None
    model_max_tokens: int = 8192
    token_counter: TokenCounter | None = None
    char_fast_estimate: float = DEFAULT_CHARS_PER_TOKEN

    def is_active(self) -> bool:
        return (
            self.strategy == "chunk"
            and self.chunker is not None
            and self.aggregator is not None
            and self.token_counter is not None
        )


def _bump(counter: Counter[str] | None, reason: str) -> None:
    if counter is not None:
        counter[reason] += 1


def _bump_and_log(
    counter: Counter[str] | None,
    reason: str,
    ci_id: str,
    *,
    warn_first: bool = False,
) -> None:
    """Bump ``counter[reason]`` and log a per-record skip line.

    When ``warn_first`` is set and this is the first time the reason hits
    for the current file (i.e. ``counter[reason] == 0`` *before* the bump),
    emit one WARNING; otherwise DEBUG. Used for anomaly reasons
    (e.g. ``missing_content_type``) where a single prominent line per
    file surfaces data-quality drift without flooding the log.
    """
    first_hit = warn_first and counter is not None and counter[reason] == 0
    _bump(counter, reason)
    if first_hit:
        log.warning(
            "skip ci_id=%s reason=%s (further occurrences at DEBUG)",
            ci_id,
            reason,
        )
    else:
        log.debug("skip ci_id=%s reason=%s", ci_id, reason)


@dataclass(frozen=True)
class EncoderConfig:
    """Runtime config threaded through the per-record / batcher calls."""

    batch_size: int
    min_char_length: int = 400
    content_types: frozenset[str] = frozenset({"ar"})
    # Long-doc handling is opt-in and defaults to None; when None or
    # inactive, the text-level path encodes one-shot and trunc-by-tokenizer
    # applies (pre-step-16 behaviour). See :class:`LongDocConfig`.
    long_doc: LongDocConfig | None = None


def build_embedder_tag(model_name: str, model_revision: str | None) -> str:
    """Formatting convention matching the legacy `embedder` field (e.g. `name@revision`)."""
    return f"{model_name}@{model_revision or 'default'}"


def _content_type_skip_reason(record: dict, cfg: EncoderConfig) -> str | None:
    """Return the filter reason to skip under, or ``None`` to pass.

    Splits the two failure modes so the per-file tally surfaces them
    separately: ``content_type`` is the routine allow-list reject
    (expected on every shard), ``missing_content_type`` is an anomaly —
    the legacy pipeline (``a433970``) treated absent ``tp`` as a reject
    too; passing it through would hide data-quality drift.
    """
    tp = record.get("tp")
    if tp is None:
        return FILTER_MISSING_CONTENT_TYPE
    if tp not in cfg.content_types:
        return FILTER_CONTENT_TYPE
    return None


def _document_text(record: dict) -> str:
    """Return the document text: ``record["ft"]`` if present, else rebuilt from ``sents``.

    Rebuilt-corpus shards (e.g. ``s3://<…>-rebuilt-final/``) carry ``ft`` directly
    and have no ``sents`` key; lingproc-enriched shards carry ``sents`` with
    token offsets. Either is a valid input.
    """
    ft = record.get("ft")
    if isinstance(ft, str) and ft:
        return ft
    return rebuild_ft_from_offsets(record.get("sents") or [])


def is_long_doc(text: str, cfg: LongDocConfig) -> bool:
    """Return True iff ``text`` exceeds ``cfg.model_max_tokens`` tokens.

    Uses a cheap character-length estimate first to skip tokenisation for
    clearly-short inputs; only calls ``cfg.token_counter`` when the char
    count is within the upper bound implied by ``char_fast_estimate``.
    Returns False when ``cfg.token_counter`` is not set (detection is
    disabled).
    """
    if cfg.token_counter is None:
        return False
    # Cheap upper bound on token count: len(text) / chars_per_token.
    # If the upper bound is still under the ceiling, we can't possibly
    # be long.
    if len(text) / max(cfg.char_fast_estimate, 1e-6) < cfg.model_max_tokens:
        return False
    return cfg.token_counter(text) > cfg.model_max_tokens


# ---------------------------------------------------------------------------
# Text level — batched across records
# ---------------------------------------------------------------------------


@dataclass
class _PendingText:
    ci_id: str
    # A short doc contributes one text; a long doc contributes K chunk
    # texts. At flush time the batcher slices the encode output back
    # per-pending and applies aggregation iff ``len(texts) > 1``.
    texts: list[str]
    ci_type: str | None
    was_chunked: bool = False

    @property
    def n_texts(self) -> int:
        return len(self.texts)


class TextBatcher:
    """Accumulates records into a buffer; flushes produce :class:`TextRecord` objects.

    When long-doc handling is active (:attr:`EncoderConfig.long_doc` with
    ``strategy="chunk"``), records whose text exceeds the model's context
    are chunked at add-time. All pending texts — short-doc singletons and
    long-doc chunk lists alike — are encoded in one batched
    :func:`encode_texts` call on flush; chunked records get their K
    vectors aggregated back into one document embedding afterward.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        cfg: EncoderConfig,
        embedder_tag: str,
        filter_counter: Counter[str] | None = None,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._embedder_tag = embedder_tag
        self._filter_counter = filter_counter
        self._pending: list[_PendingText] = []
        self._total_texts = 0  # sum of len(p.texts) across pending items

    def __len__(self) -> int:
        return len(self._pending)

    def add(self, record: dict) -> list[TextRecord]:
        """Add one record. Returns emitted records if a flush happened, else []."""
        reason = _content_type_skip_reason(record, self._cfg)
        if reason is not None:
            _bump_and_log(
                self._filter_counter,
                reason,
                record.get("id") or "<no-id>",
                warn_first=(reason == FILTER_MISSING_CONTENT_TYPE),
            )
            return []
        ci_id = record.get("id")
        if not ci_id:
            _bump(self._filter_counter, FILTER_MISSING_ID)
            log.debug("skip ci_id=<no-id> reason=%s", FILTER_MISSING_ID)
            return []
        text = _document_text(record)
        if len(text) <= self._cfg.min_char_length:
            _bump(self._filter_counter, FILTER_TOO_SHORT)
            log.debug("skip ci_id=%s reason=%s", ci_id, FILTER_TOO_SHORT)
            return []

        texts, was_chunked = self._maybe_chunk(ci_id, text)
        if was_chunked:
            _bump(self._filter_counter, LONG_DOC_CHUNKED)

        self._pending.append(
            _PendingText(
                ci_id=ci_id,
                texts=texts,
                ci_type=record.get("tp"),
                was_chunked=was_chunked,
            )
        )
        self._total_texts += len(texts)
        if self._total_texts >= self._cfg.batch_size:
            return self.flush()
        return []

    def _maybe_chunk(self, ci_id: str, text: str) -> tuple[list[str], bool]:
        """Return ``(texts, was_chunked)`` for one record.

        Short path (no long-doc handling, or doc fits in context): single
        text, ``was_chunked=False``. Long path: K chunk texts,
        ``was_chunked=True``. If the chunker returns nothing (pathological),
        falls back to the original text with a warning.
        """
        long = self._cfg.long_doc
        if long is None or not long.is_active():
            return [text], False
        if not is_long_doc(text, long):
            return [text], False
        assert long.chunker is not None  # is_active() guarantees
        chunks = long.chunker.chunk(text)
        if not chunks:
            log.warning(
                "ci_id=%s: chunker returned no chunks for long document; "
                "falling back to one-shot (will be tokenizer-truncated)",
                ci_id,
            )
            return [text], False
        texts = [c.text for c in chunks if c.text]
        if not texts:
            log.warning(
                "ci_id=%s: chunker returned only empty chunks; falling back to one-shot",
                ci_id,
            )
            return [text], False
        log.debug(
            "ci_id=%s: long-doc chunked into %d chunks",
            ci_id,
            len(texts),
        )
        return texts, True

    def flush(self) -> list[TextRecord]:
        if not self._pending:
            return []
        all_texts = [t for p in self._pending for t in p.texts]
        vecs = encode_texts(
            self._model,
            all_texts,
            batch_size=self._cfg.batch_size,
        )
        ts = utc_timestamp()
        out: list[TextRecord] = []
        offset = 0
        for p in self._pending:
            n = p.n_texts
            piece = vecs[offset : offset + n]
            offset += n
            emb = self._collapse(piece, p)
            emb_list = emb.tolist()
            out.append(
                TextRecord(
                    ci_id=p.ci_id,
                    model_id=self._embedder_tag,
                    embedding=emb_list,
                    size=len(emb_list),
                    ts=ts,
                    ci_type=p.ci_type,
                )
            )
        self._pending.clear()
        self._total_texts = 0
        return out

    def _collapse(self, piece: np.ndarray, p: _PendingText) -> np.ndarray:
        """Return the single vector for pending item ``p``.

        Short doc (one text): pass through. Long doc (K texts): aggregate
        via the configured aggregator. The aggregator contract is that
        output is a 1-D ``[D]`` unit vector.
        """
        if p.n_texts == 1:
            return piece[0]
        assert self._cfg.long_doc is not None
        aggregator = self._cfg.long_doc.aggregator
        assert aggregator is not None  # LongDocConfig.is_active() guarantees
        return aggregator.aggregate(piece)


# ---------------------------------------------------------------------------
# Sentence level — one record at a time, many sentences per call
# ---------------------------------------------------------------------------


def embed_sentence_record(
    record: dict,
    model: SentenceTransformer,
    cfg: EncoderConfig,
    filter_counter: Counter[str] | None = None,
) -> SentenceRecord | None:
    reason = _content_type_skip_reason(record, cfg)
    if reason is not None:
        _bump_and_log(
            filter_counter,
            reason,
            record.get("id") or "<no-id>",
            warn_first=(reason == FILTER_MISSING_CONTENT_TYPE),
        )
        return None

    ci_id = record.get("id")
    if not ci_id:
        _bump(filter_counter, FILTER_MISSING_ID)
        log.debug("skip ci_id=<no-id> reason=%s", FILTER_MISSING_ID)
        return None

    sents = record.get("sents") or []
    if not sents:
        _bump(filter_counter, FILTER_NO_SENTENCES)
        log.debug("skip ci_id=%s reason=%s", ci_id, FILTER_NO_SENTENCES)
        return None

    kept: list[tuple[int, str, int | None]] = []
    for idx, s in enumerate(sents):
        txt = rebuild_sentence_from_offsets(s).strip()
        if len(txt) <= cfg.min_char_length:
            continue
        offset = s.get("o") if isinstance(s, dict) else None
        kept.append((idx, txt, offset))
    if not kept:
        _bump(filter_counter, FILTER_SENTENCE_TOO_SHORT)
        log.debug("skip ci_id=%s reason=%s", ci_id, FILTER_SENTENCE_TOO_SHORT)
        return None

    texts = [t for _, t, _ in kept]
    vecs = encode_texts(
        model,
        texts,
        batch_size=cfg.batch_size,
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
    filter_counter: Counter[str] | None = None,
) -> ChunkRecord | None:
    reason = _content_type_skip_reason(record, cfg)
    if reason is not None:
        _bump_and_log(
            filter_counter,
            reason,
            record.get("id") or "<no-id>",
            warn_first=(reason == FILTER_MISSING_CONTENT_TYPE),
        )
        return None

    ci_id = record.get("id")
    if not ci_id:
        _bump(filter_counter, FILTER_MISSING_ID)
        log.debug("skip ci_id=<no-id> reason=%s", FILTER_MISSING_ID)
        return None

    text = _document_text(record)
    if len(text) <= cfg.min_char_length:
        _bump(filter_counter, FILTER_TOO_SHORT)
        log.debug("skip ci_id=%s reason=%s", ci_id, FILTER_TOO_SHORT)
        return None

    chunks = chunker.chunk(text)
    if not chunks:
        _bump(filter_counter, FILTER_NO_CHUNKS)
        log.debug("skip ci_id=%s reason=%s", ci_id, FILTER_NO_CHUNKS)
        return None

    texts = [c.text for c in chunks]
    vecs = encode_texts(
        model,
        texts,
        batch_size=cfg.batch_size,
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
    filter_counter: Counter[str] | None = None,
) -> Iterable[dict]:
    """Yield output dicts (already `.to_dict()`) for all input records.

    Callers pick ``level`` and pass the corresponding helpers. ``chunker`` is
    required when ``level == "chunk"``. Pass ``filter_counter`` to collect a
    per-reason tally of filtered records; each level's filter branch bumps the
    matching :data:`FILTER_*` key.
    """
    if level == "text":
        batcher = TextBatcher(model, cfg, embedder_tag, filter_counter=filter_counter)
        for rec in records:
            for out in batcher.add(rec):
                yield out.to_dict()
        for out in batcher.flush():
            yield out.to_dict()
    elif level == "sentence":
        for rec in records:
            out = embed_sentence_record(rec, model, cfg, filter_counter=filter_counter)
            if out is not None:
                yield out.to_dict()
    elif level == "chunk":
        if chunker is None:
            raise ValueError("chunking strategy required for level='chunk'")
        for rec in records:
            out = embed_chunk_record(
                rec, model, cfg, chunker, filter_counter=filter_counter
            )
            if out is not None:
                yield out.to_dict()
    else:
        raise ValueError(f"unknown embedding level: {level!r}")
