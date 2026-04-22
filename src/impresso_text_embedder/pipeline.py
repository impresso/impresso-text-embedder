"""Per-file orchestration: read input from S3, embed, write output, upload."""

from __future__ import annotations

import bz2
import json
import logging
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from impresso_text_embedder import io as s3io
from impresso_text_embedder.embed import EncoderConfig, build_embedder_tag, embed_records

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from impresso_text_embedder.chunking.base import ChunkingStrategy

log = logging.getLogger(__name__)

DEFAULT_MODEL_SLUG_PREFIX_TO_STRIP = "Alibaba-NLP/"


def model_slug(model_name: str) -> str:
    """Strip the vendor prefix for the S3 output path."""
    if model_name.startswith(DEFAULT_MODEL_SLUG_PREFIX_TO_STRIP):
        return model_name[len(DEFAULT_MODEL_SLUG_PREFIX_TO_STRIP) :]
    return model_name.split("/")[-1]


@dataclass(frozen=True)
class PipelineConfig:
    input_bucket: str
    output_bucket: str
    input_prefix: str
    model_name: str
    model_revision: str | None
    level: str
    chunking_strategy_name: str
    force: bool
    encoder: EncoderConfig


def iter_input_lines(bucket: str, key: str) -> Iterator[str]:
    """Yield raw JSON lines streamed from an S3 `.jsonl.bz2` object."""
    return s3io.iter_jsonl_bz2(bucket, key)


def _iter_parsed(lines: Iterable[str]) -> Iterator[dict]:
    for i, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("malformed JSON at line %d: %s", i + 1, exc)


def _resolve_chunker(cfg: PipelineConfig) -> ChunkingStrategy | None:
    if cfg.level != "chunk":
        return None
    from impresso_text_embedder.chunking import get_strategy

    return get_strategy(cfg.chunking_strategy_name)


def process_file(
    input_key: s3io.InputKey,
    model: SentenceTransformer,
    cfg: PipelineConfig,
    chunker: ChunkingStrategy | None = None,
) -> bool:
    """Process one ``(provider, alias, year)`` input file end-to-end.

    Returns True if the file was (re)processed, False if skipped because the
    output already exists and ``cfg.force`` is False.
    """
    slug = model_slug(cfg.model_name)
    output_key = s3io.build_output_key(
        provider=input_key.provider,
        alias=input_key.alias,
        year=input_key.year,
        model_slug=slug,
    )

    if not cfg.force and s3io.object_exists(cfg.output_bucket, output_key):
        log.info(
            "skip s3://%s/%s (output exists; pass --force to overwrite)",
            cfg.output_bucket,
            output_key,
        )
        return False

    embedder_tag = build_embedder_tag(cfg.model_name, cfg.model_revision)
    if chunker is None:
        chunker = _resolve_chunker(cfg)

    log.info(
        "process s3://%s/%s -> s3://%s/%s (level=%s)",
        cfg.input_bucket,
        input_key.key,
        cfg.output_bucket,
        output_key,
        cfg.level,
    )

    records = _iter_parsed(iter_input_lines(cfg.input_bucket, input_key.key))

    with tempfile.NamedTemporaryFile(
        prefix="impresso-embed-", suffix=".jsonl.bz2", delete=False
    ) as tmp:
        local_path = Path(tmp.name)

    written = 0
    try:
        with bz2.open(local_path, "wt", encoding="utf-8") as fh:
            for out in embed_records(
                records,
                level=cfg.level,
                model=model,
                cfg=cfg.encoder,
                embedder_tag=embedder_tag,
                chunker=chunker,
            ):
                fh.write(json.dumps(out, ensure_ascii=False))
                fh.write("\n")
                written += 1
        s3io.upload_local_file(local_path, cfg.output_bucket, output_key)
    finally:
        try:
            local_path.unlink()
        except FileNotFoundError:
            pass

    log.info("wrote %d records to s3://%s/%s", written, cfg.output_bucket, output_key)
    return True


def process_provider(
    provider: str,
    model: SentenceTransformer,
    cfg: PipelineConfig,
    alias_filter: set[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Walk a provider's inputs and process every matching file.

    Returns a summary dict: ``{"processed": n, "skipped": n, "files": [list]}``.
    """
    summary = {"processed": 0, "skipped": 0, "files": []}
    chunker = _resolve_chunker(cfg)

    for input_key in s3io.list_input_keys(
        bucket=cfg.input_bucket,
        provider=provider,
        input_prefix=cfg.input_prefix,
        alias_filter=alias_filter,
        year_min=year_min,
        year_max=year_max,
    ):
        summary["files"].append(input_key.key)
        if dry_run:
            output_key = s3io.build_output_key(
                input_key.provider, input_key.alias, input_key.year, model_slug(cfg.model_name)
            )
            exists = (not cfg.force) and s3io.object_exists(cfg.output_bucket, output_key)
            if exists:
                summary["skipped"] += 1
                log.info("[dry-run] would skip %s", input_key.key)
            else:
                summary["processed"] += 1
                log.info("[dry-run] would process %s", input_key.key)
            continue

        did = process_file(input_key, model, cfg, chunker=chunker)
        if did:
            summary["processed"] += 1
        else:
            summary["skipped"] += 1

    return summary


def with_batch_size(cfg: PipelineConfig, batch_size: int) -> PipelineConfig:
    """Return a copy of ``cfg`` with a different encoder batch size."""
    return replace(cfg, encoder=replace(cfg.encoder, batch_size=batch_size))
