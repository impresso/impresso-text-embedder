"""Per-file orchestration: read input from S3, embed, write output, upload.

The per-provider loop overlaps three stages via single-slot executors so the
GPU is not stalled by IO between files:

* a **prefetch** thread downloads the next file's ``.jsonl.bz2`` to a local
  tempfile (multipart ranged GETs via :data:`io.DEFAULT_TRANSFER_CONFIG`);
* the main thread encodes the current file from its already-downloaded
  local tempfile;
* an **upload** thread sends the completed output tempfile to S3 while the
  main thread moves on to the next file.

Design and rationale: ``.progress/io-throughput/notes.md``.
"""

from __future__ import annotations

import bz2
import itertools
import logging
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
from tqdm import tqdm

from impresso_text_embedder import io as s3io
from impresso_text_embedder.embed import EncoderConfig, build_embedder_tag, embed_records
from impresso_text_embedder.telemetry import (
    GpuSampler,
    GpuSummary,
    StageTimer,
    format_stats_line,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from impresso_text_embedder.chunking.base import ChunkingStrategy

log = logging.getLogger(__name__)

DEFAULT_MODEL_SLUG_PREFIX_TO_STRIP = "Alibaba-NLP/"

# Impresso-convention slugs for models we ship with. Used as the
# ``<model-slug>`` segment of the output S3 key. Unknown models fall back to
# stripping the HF vendor prefix (or the last path component).
MODEL_SLUG_OVERRIDES: dict[str, str] = {
    "Alibaba-NLP/gte-multilingual-base": "embeddings_gte_v1-1-0",
}


def model_slug(model_name: str) -> str:
    """Return the Impresso-convention slug for the S3 output path."""
    if model_name in MODEL_SLUG_OVERRIDES:
        return MODEL_SLUG_OVERRIDES[model_name]
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


def _iter_parsed(lines: Iterable[str]) -> Iterator[dict]:
    for i, line in enumerate(lines):
        try:
            yield orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            log.warning("malformed JSON at line %d: %s", i + 1, exc)


def _resolve_chunker(cfg: PipelineConfig) -> ChunkingStrategy | None:
    if cfg.level != "chunk":
        return None
    from impresso_text_embedder.chunking import get_strategy

    return get_strategy(cfg.chunking_strategy_name)


def _should_skip(
    input_key: s3io.InputKey,
    output_bucket: str,
    output_key: str,
) -> tuple[bool, str]:
    """Decide whether an existing output is still fresh relative to its input.

    Returns ``(skip, reason)``. ``reason`` is a short string suitable for logging
    and is meaningful whether ``skip`` is True or False.
    """
    output_lm = s3io.head_last_modified(output_bucket, output_key)
    if output_lm is None:
        return False, "output missing"
    # Without an input timestamp (e.g. manually-built InputKey in tests) keep the
    # old "output exists ⇒ skip" semantics.
    if input_key.last_modified is None:
        return True, "output exists (no input timestamp)"
    if output_lm >= input_key.last_modified:
        return True, f"output up-to-date (input={input_key.last_modified!s} output={output_lm!s})"
    return False, f"input newer than output (input={input_key.last_modified!s} output={output_lm!s})"


def _make_tempfile(suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(prefix="impresso-embed-", suffix=suffix, delete=False) as tmp:
        return Path(tmp.name)


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _encode_to_local(
    input_key: s3io.InputKey,
    local_input: Path,
    local_output: Path,
    model: SentenceTransformer,
    cfg: PipelineConfig,
    chunker: ChunkingStrategy | None,
    timer: StageTimer,
) -> tuple[int, Counter[str]]:
    """Decode records from ``local_input`` and write embeddings to ``local_output``.

    Returns ``(written, filter_counter)``: the number of records written and a
    per-reason tally of records that were filtered out (keys are the
    :data:`embed.FILTER_*` constants).
    """
    embedder_tag = build_embedder_tag(cfg.model_name, cfg.model_revision)
    records = _iter_parsed(s3io.iter_jsonl_bz2_path(local_input))
    written = 0
    filter_counter: Counter[str] = Counter()
    with timer.stage("encode"), GpuSampler() as gpu:
        with bz2.open(local_output, "wb") as fh:
            for out in embed_records(
                records,
                level=cfg.level,
                model=model,
                cfg=cfg.encoder,
                embedder_tag=embedder_tag,
                chunker=chunker,
                filter_counter=filter_counter,
            ):
                fh.write(orjson.dumps(out, option=orjson.OPT_APPEND_NEWLINE))
                written += 1
    _LAST_GPU_SUMMARY[input_key.key] = gpu.summary()
    return written, filter_counter


# Keyed by input-key.key; the per-file GPU summary is published here by
# ``_encode_to_local`` and consumed by ``process_provider`` when it builds
# the per-file log line. This avoids threading an extra return value through
# every helper.
_LAST_GPU_SUMMARY: dict[str, GpuSummary] = {}


def process_file(
    input_key: s3io.InputKey,
    model: SentenceTransformer,
    cfg: PipelineConfig,
    chunker: ChunkingStrategy | None = None,
) -> bool:
    """Process one ``(provider, alias, year)`` input file end-to-end.

    Sequential path: download → encode → upload. ``process_provider`` uses a
    pipelined variant that overlaps these stages across files.

    Returns True if the file was (re)processed, False if skipped.
    """
    slug = model_slug(cfg.model_name)
    output_key = s3io.build_output_key(
        provider=input_key.provider,
        alias=input_key.alias,
        year=input_key.year,
        model_slug=slug,
    )

    if not cfg.force:
        skip, reason = _should_skip(input_key, cfg.output_bucket, output_key)
        if skip:
            log.info(
                "skip s3://%s/%s (%s; pass --force to overwrite)",
                cfg.output_bucket,
                output_key,
                reason,
            )
            return False
        if reason != "output missing":
            log.info(
                "reprocess s3://%s/%s (%s)", cfg.output_bucket, output_key, reason
            )

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

    local_input = _make_tempfile(".jsonl.bz2")
    local_output = _make_tempfile(".jsonl.bz2")
    timer = StageTimer()
    try:
        with timer.stage("download"):
            s3io.download_to_local(cfg.input_bucket, input_key.key, local_input)
        written, filter_counter = _encode_to_local(
            input_key, local_input, local_output, model, cfg, chunker, timer
        )
        with timer.stage("upload"):
            s3io.upload_local_file(local_output, cfg.output_bucket, output_key)
    finally:
        _unlink_quiet(local_input)
        _unlink_quiet(local_output)

    gpu = _LAST_GPU_SUMMARY.pop(input_key.key, GpuSummary())
    log.info(
        format_stats_line(
            f"done s3://{cfg.output_bucket}/{output_key}",
            timer,
            records=written,
            gpu=gpu,
            filter_counter=filter_counter,
        )
    )
    return True


def _plan_files(
    provider: str,
    cfg: PipelineConfig,
    alias_filter: set[str] | None,
    year_min: int | None,
    year_max: int | None,
    limit: int | None = None,
) -> tuple[list[tuple[s3io.InputKey, str]], list[str]]:
    """Return ``(to_process, skipped)``.

    ``to_process`` is a list of ``(input_key, output_key)`` pairs that passed
    the skip check (or bypassed it because ``cfg.force``). ``skipped`` is a
    list of input keys that were skipped; they are still recorded in the
    run summary.
    """
    slug = model_slug(cfg.model_name)
    to_process: list[tuple[s3io.InputKey, str]] = []
    skipped: list[str] = []
    listing: Iterable[s3io.InputKey] = s3io.list_input_keys(
        bucket=cfg.input_bucket,
        provider=provider,
        input_prefix=cfg.input_prefix,
        alias_filter=alias_filter,
        year_min=year_min,
        year_max=year_max,
    )
    if limit is not None:
        log.info("limit=%d applied; listing truncated", limit)
        listing = itertools.islice(listing, limit)
    for input_key in listing:
        output_key = s3io.build_output_key(
            input_key.provider, input_key.alias, input_key.year, slug
        )
        if cfg.force:
            to_process.append((input_key, output_key))
            continue
        skip, reason = _should_skip(input_key, cfg.output_bucket, output_key)
        if skip:
            log.info(
                "skip s3://%s/%s (%s; pass --force to overwrite)",
                cfg.output_bucket,
                output_key,
                reason,
            )
            skipped.append(input_key.key)
            continue
        if reason != "output missing":
            log.info(
                "reprocess s3://%s/%s (%s)", cfg.output_bucket, output_key, reason
            )
        to_process.append((input_key, output_key))
    return to_process, skipped


def _dry_run_summary(
    provider: str,
    cfg: PipelineConfig,
    alias_filter: set[str] | None,
    year_min: int | None,
    year_max: int | None,
    limit: int | None = None,
) -> dict:
    summary: dict = {"processed": 0, "skipped": 0, "files": []}
    slug = model_slug(cfg.model_name)
    listing: Iterable[s3io.InputKey] = s3io.list_input_keys(
        bucket=cfg.input_bucket,
        provider=provider,
        input_prefix=cfg.input_prefix,
        alias_filter=alias_filter,
        year_min=year_min,
        year_max=year_max,
    )
    if limit is not None:
        log.info("limit=%d applied; listing truncated", limit)
        listing = itertools.islice(listing, limit)
    for input_key in listing:
        summary["files"].append(input_key.key)
        output_key = s3io.build_output_key(
            input_key.provider, input_key.alias, input_key.year, slug
        )
        if cfg.force:
            summary["processed"] += 1
            log.info("[dry-run] would process %s (forced)", input_key.key)
            continue
        skip, reason = _should_skip(input_key, cfg.output_bucket, output_key)
        if skip:
            summary["skipped"] += 1
            log.info("[dry-run] would skip %s (%s)", input_key.key, reason)
        else:
            summary["processed"] += 1
            log.info("[dry-run] would process %s (%s)", input_key.key, reason)
    return summary


def process_provider(
    provider: str,
    model: SentenceTransformer,
    cfg: PipelineConfig,
    alias_filter: set[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Walk a provider's inputs and process every matching file.

    Overlaps download/encode/upload via two single-slot thread pools so the
    GPU can keep encoding while the next input is downloading and the
    previous output is uploading.

    ``limit`` truncates the S3 listing after N keys (lexicographic key
    order). Applied before the skip-if-exists check, so already-processed
    files count toward the limit.

    Returns a summary dict: ``{"processed": n, "skipped": n, "files": [list]}``.
    """
    if dry_run:
        return _dry_run_summary(provider, cfg, alias_filter, year_min, year_max, limit)

    to_process, skipped_keys = _plan_files(
        provider, cfg, alias_filter, year_min, year_max, limit
    )
    summary: dict = {
        "processed": 0,
        "skipped": len(skipped_keys),
        "files": list(skipped_keys),
    }
    if not to_process:
        return summary

    chunker = _resolve_chunker(cfg)

    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as prefetcher,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="upload") as uploader,
        tqdm(
            total=len(to_process),
            file=sys.stderr,
            disable=None,  # auto-disable on non-tty (tests, piped output)
            desc=provider,
            unit="file",
        ) as pbar,
    ):
        # Prime the prefetch pipeline with the first file.
        next_input = _make_tempfile(".jsonl.bz2")
        next_fetch: Future = prefetcher.submit(
            s3io.download_to_local, cfg.input_bucket, to_process[0][0].key, next_input
        )
        prev_upload: Future | None = None
        prev_output: Path | None = None

        for i, (input_key, output_key) in enumerate(to_process):
            summary["files"].append(input_key.key)
            pbar.set_description(f"{input_key.alias}/{input_key.year}")
            local_input = next_input
            timer = StageTimer()
            try:
                with timer.stage("download"):
                    next_fetch.result()

                # Kick off prefetch of file i+1 while we encode file i.
                if i + 1 < len(to_process):
                    next_input = _make_tempfile(".jsonl.bz2")
                    next_key = to_process[i + 1][0].key
                    next_fetch = prefetcher.submit(
                        s3io.download_to_local, cfg.input_bucket, next_key, next_input
                    )

                local_output = _make_tempfile(".jsonl.bz2")
                try:
                    written, filter_counter = _encode_to_local(
                        input_key, local_input, local_output, model, cfg, chunker, timer
                    )
                except Exception:
                    _unlink_quiet(local_output)
                    raise

                # Wait for the previous upload before starting a new one
                # (single-slot uploader + fail-fast on prior error).
                if prev_upload is not None:
                    with timer.stage("upload_wait"):
                        prev_upload.result()
                    if prev_output is not None:
                        _unlink_quiet(prev_output)

                prev_upload = uploader.submit(
                    s3io.upload_local_file, local_output, cfg.output_bucket, output_key
                )
                prev_output = local_output
            finally:
                _unlink_quiet(local_input)

            summary["processed"] += 1
            gpu = _LAST_GPU_SUMMARY.pop(input_key.key, GpuSummary())
            log.info(
                format_stats_line(
                    f"done s3://{cfg.output_bucket}/{output_key}",
                    timer,
                    records=written,
                    gpu=gpu,
                    filter_counter=filter_counter,
                )
            )
            pbar.set_postfix(**_postfix_stats(timer, gpu))
            pbar.update(1)

        # Drain: wait for the final upload and clean up its tempfile.
        if prev_upload is not None:
            prev_upload.result()
            if prev_output is not None:
                _unlink_quiet(prev_output)

    return summary


def _postfix_stats(timer: StageTimer, gpu: GpuSummary) -> dict[str, str]:
    """Short postfix dict for the tqdm bar: `dl=…s enc=…s up=…s gpu=…%`."""
    totals = timer.totals
    out: dict[str, str] = {}
    if "download" in totals:
        out["dl"] = f"{totals['download']:.1f}s"
    if "encode" in totals:
        out["enc"] = f"{totals['encode']:.1f}s"
    if "upload_wait" in totals:
        out["up"] = f"{totals['upload_wait']:.1f}s"
    if gpu.samples > 0:
        out["gpu"] = f"{gpu.mean:.0f}%"
    return out


def with_batch_size(cfg: PipelineConfig, batch_size: int) -> PipelineConfig:
    """Return a copy of ``cfg`` with a different encoder batch size."""
    return replace(cfg, encoder=replace(cfg.encoder, batch_size=batch_size))
