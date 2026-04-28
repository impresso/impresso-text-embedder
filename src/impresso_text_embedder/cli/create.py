"""Entry point for ``impresso-embed-create``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from impresso_text_embedder.embed import EncoderConfig, LongDocConfig
from impresso_text_embedder.logging_setup import configure_logging
from impresso_text_embedder.model import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION
from impresso_text_embedder.pipeline import PipelineConfig, process_provider

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-embed-create",
        description=(
            "Embed .jsonl.bz2 content-item shards under a given provider on S3, "
            "writing one output .jsonl.bz2 per input file."
        ),
    )
    p.add_argument("--provider", required=True, help="provider code (e.g. SNL)")
    p.add_argument(
        "--input-bucket",
        default="122-rebuilt-final",
        help="S3 bucket holding input .jsonl.bz2 shards under <provider>/<alias>/",
    )
    p.add_argument(
        "--output-bucket",
        default="140-processed-data-sandbox",
        help="S3 bucket receiving embedding outputs; keys mirror input under embeddings/docs/<model-slug>/",
    )
    p.add_argument("--input-prefix", default="", help="optional prefix inside the input bucket")

    p.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model id to load via sentence-transformers",
    )
    p.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="HuggingFace revision (commit/tag/branch) to pin the model to; not appended to the output slug",
    )

    p.add_argument(
        "--embedding-level",
        choices=["text", "sentence", "chunk"],
        default="text",
        help="granularity of the produced embeddings: one per document, sentence, or chunk",
    )
    p.add_argument(
        "--chunking-strategy",
        default="semantic",
        help="only used when --embedding-level=chunk",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch size for encoder.encode(); when omitted, picked per detected GPU profile",
    )
    p.add_argument(
        "--min-char-length",
        type=int,
        default=800,
        help="skip records whose reconstructed text is shorter than this many characters",
    )
    p.add_argument(
        "--content-type",
        nargs="+",
        default=["ar"],
        choices=["ar", "page"],
        help="keep only records whose 'tp' field is in this allow-list",
    )
    # Long-document handling at --embedding-level=text. Partial landing:
    # only the ``chunk`` strategy with ``mean`` aggregation is wired today.
    # See .progress/long-doc-chunking/notes.md for the full design space
    # and the list of deferred strategies.
    p.add_argument(
        "--long-doc-strategy",
        choices=["truncate", "chunk"],
        default="chunk",
        help=(
            "how to handle documents longer than the model's max context at "
            "--embedding-level=text. 'chunk' (default): split into ≤"
            "--long-doc-chunk-tokens chunks, encode each, and aggregate via "
            "--long-doc-aggregation. 'truncate': let the tokenizer drop the "
            "tail (legacy behaviour, pre-step-16). Only "
            "--long-doc-aggregation=mean is implemented today."
        ),
    )
    p.add_argument(
        "--long-doc-chunk-tokens",
        type=int,
        default=None,
        help=(
            "max tokens per chunk when --long-doc-strategy=chunk. When "
            "omitted (default), auto-derived from the model tokenizer as "
            "`model_max_length - num_special_tokens_to_add(pair=False)` — "
            "i.e. the full model window minus CLS/SEP headroom (8190 for "
            "gte-multilingual-base). Pass an explicit integer to override."
        ),
    )
    p.add_argument(
        "--long-doc-aggregation",
        # Only 'mean' is landed; additional strategies (max, length-weighted,
        # first-chunk, …) are registered in impresso_text_embedder.aggregation
        # as they ship.
        choices=["mean"],
        default="mean",
        help="how to combine chunk embeddings into one document vector when --long-doc-strategy=chunk",
    )

    # Numerical-ablation toggles. Defaults preserve the historical fast path
    # (bf16 autocast + xformers memory_efficient_attention + unpad_inputs);
    # flipping any of them lets an operator bisect drift against an older
    # baseline by re-encoding a small slice with one lever changed at a time.
    p.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default="bf16",
        help=(
            "encode-time numerical precision. 'bf16' (default) wraps "
            "model.encode() in a torch.autocast(dtype=bfloat16) scope on "
            "CUDA; 'fp32' skips that scope (model weights are already fp32). "
            "Use 'fp32' to isolate bf16 as a drift source."
        ),
    )
    p.add_argument(
        "--attention",
        choices=["xformers", "eager"],
        default="xformers",
        help=(
            "attention kernel selection. 'xformers' (default) sets "
            "use_memory_efficient_attention=True on the model config — "
            "Alibaba's modeling file then routes through "
            "xformers.ops.memory_efficient_attention. 'eager' omits the "
            "flag, leaving the default attention path. 'xformers' requires "
            "CUDA and an importable xformers package; otherwise model load "
            "fails fast."
        ),
    )
    p.add_argument(
        "--unpad-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "set unpad_inputs=True on the model config (default). Use "
            "--no-unpad-inputs to disable; combinable with --attention=eager "
            "for ablation against an older padded baseline."
        ),
    )

    p.add_argument("--alias", nargs="*", default=None, help="filter to these aliases")
    p.add_argument(
        "--year-min", type=int, default=None, help="skip shards whose year is strictly below this"
    )
    p.add_argument(
        "--year-max", type=int, default=None, help="skip shards whose year is strictly above this"
    )

    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="reprocess even if output exists on S3",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="list files and whether they'd be processed; no model loaded, no work done",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "process at most the first N listed shards (lexicographic S3 key order), "
            "applied before the skip-if-exists check; handy for smoke tests"
        ),
    )

    # Multi-GPU horizontal sharding (step 18). One runai job per shard, one
    # GPU per job, model replicated across jobs. Defaults reproduce the
    # single-job path bit-for-bit. Both flags must be set together.
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=(
            "zero-indexed shard this job owns (use with --num-shards). "
            "Round-robin over list_objects_v2 lexicographic order; "
            "see .progress/multi-gpu-sharding/notes.md."
        ),
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total number of shards across the multi-job run (default 1 = no sharding)",
    )

    p.add_argument(
        "--log-level-file",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="level for the per-provider log file (terminal is always ERROR-only)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "override the base log directory; the full path becomes "
            "<log-dir>/<YYYY-MM-DD>/<provider>.log. Default is "
            "/rcp-scratch/<username>/experiments/embeddings (requires the PVC mounted)"
        ),
    )
    return p


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in dependencies
        return
    load_dotenv()


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    encoder = EncoderConfig(
        batch_size=args.batch_size,
        min_char_length=args.min_char_length,
        content_types=frozenset(args.content_type),
        precision=args.precision,
    )
    return PipelineConfig(
        input_bucket=args.input_bucket,
        output_bucket=args.output_bucket,
        input_prefix=args.input_prefix,
        model_name=args.model_name,
        model_revision=args.model_revision,
        level=args.embedding_level,
        chunking_strategy_name=args.chunking_strategy,
        force=args.force,
        encoder=encoder,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


def _build_long_doc_config(args: argparse.Namespace, model: object) -> LongDocConfig:
    """Construct the long-doc config using the loaded model's tokenizer.

    Kept out of :func:`_build_pipeline_config` because that runs before
    the model loads (dry-run path); the tokenizer-derived token counter
    only exists post-load. Returns a fully-populated :class:`LongDocConfig`
    ready to drop into ``EncoderConfig.long_doc``.
    """
    from impresso_text_embedder.aggregation import get_strategy as get_aggregator
    from impresso_text_embedder.chunking import get_strategy as get_chunker

    tokenizer = model.tokenizer  # type: ignore[attr-defined]
    # Model's native max context. Coerced defensively so a model whose
    # attribute is unset, None, or a Mock (tests) falls back cleanly to
    # a sensible fallback instead of crashing the log format.
    raw_max = getattr(model, "max_seq_length", None)
    try:
        model_max = int(raw_max) if raw_max is not None else _FALLBACK_CHUNK_TOKENS
    except (TypeError, ValueError):
        model_max = _FALLBACK_CHUNK_TOKENS

    chunk_tokens = _resolve_chunk_tokens(args.long_doc_chunk_tokens, tokenizer)

    def _count(text: str) -> int:
        # verbose=False silences transformers' "sequence length > model_max_length"
        # warning — we count to decide whether to chunk, never feed the raw ids
        # to the model.
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))

    chunker = get_chunker(
        "fixed-window",
        tokenizer=tokenizer,
        max_tokens=chunk_tokens,
    )
    aggregator = get_aggregator(args.long_doc_aggregation)
    return LongDocConfig(
        strategy="chunk",
        chunker=chunker,
        aggregator=aggregator,
        model_max_tokens=model_max,
        token_counter=_count,
    )


# Legacy fallback used when neither the model nor the tokenizer advertises
# a usable max-sequence length (stubs in tests, exotic non-HF tokenizers).
# Historical value from the pre-auto-derivation default.
_FALLBACK_CHUNK_TOKENS = 8000


def _resolve_chunk_tokens(cli_value: int | None, tokenizer: object) -> int:
    """Pick the chunk-token ceiling.

    Precedence: explicit CLI value > tokenizer-derived value > fallback.
    Tokenizer-derived = ``model_max_length - num_special_tokens_to_add(pair=False)``,
    which matches the HF-canonical way to size a window that will fit after
    the encoder prepends CLS/SEP.
    """
    if cli_value is not None:
        return cli_value
    model_max_length = getattr(tokenizer, "model_max_length", None)
    special = getattr(tokenizer, "num_special_tokens_to_add", None)
    try:
        if model_max_length is None or not callable(special):
            return _FALLBACK_CHUNK_TOKENS
        derived = int(model_max_length) - int(special(pair=False))
        if derived <= 0:
            return _FALLBACK_CHUNK_TOKENS
        return derived
    except (TypeError, ValueError):
        return _FALLBACK_CHUNK_TOKENS


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error(f"--limit must be >= 0, got {args.limit}")
    if args.num_shards < 1:
        parser.error(f"--num-shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(
            f"--shard-index must be in [0, --num-shards), got "
            f"--shard-index={args.shard_index} --num-shards={args.num_shards}"
        )
    # Foot-gun: --num-shards 4 alone would silently run only shard 0 of 4.
    # Require both flags or neither.
    if (args.shard_index != 0) != (args.num_shards != 1):
        parser.error(
            "--shard-index and --num-shards must be set together; got "
            f"--shard-index={args.shard_index} --num-shards={args.num_shards}"
        )
    log_path = configure_logging(
        provider=args.provider,
        log_dir=args.log_dir,
        log_level_file=args.log_level_file,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(f"logging to {log_path}", file=sys.stderr)
    _load_env()

    if args.batch_size is None:
        from impresso_text_embedder.accel import detect_profile

        args.batch_size = detect_profile().default_batch_size

    cfg = _build_pipeline_config(args)

    if args.dry_run:
        model = None  # not used
    else:
        from impresso_text_embedder.model import load_model

        use_xformers = args.attention == "xformers"
        model = load_model(
            name=args.model_name,
            revision=args.model_revision,
            use_xformers=use_xformers,
            unpad_inputs=args.unpad_inputs,
        )
        log.info(
            "ablation toggles: precision=%s attention=%s unpad_inputs=%s",
            args.precision,
            args.attention,
            args.unpad_inputs,
        )
        if args.long_doc_strategy == "chunk":
            long_doc = _build_long_doc_config(args, model)
            cfg = dataclasses.replace(
                cfg,
                encoder=dataclasses.replace(cfg.encoder, long_doc=long_doc),
            )
            # Resolve once for the log line; the same resolution happens
            # inside _build_long_doc_config, so the value here matches what
            # the chunker actually got.
            resolved_chunk_tokens = _resolve_chunk_tokens(
                args.long_doc_chunk_tokens, model.tokenizer
            )
            log.info(
                "long-doc handling active: strategy=chunk chunker=fixed-window "
                "chunk_tokens=%d aggregation=%s model_max_tokens=%d",
                resolved_chunk_tokens,
                args.long_doc_aggregation,
                long_doc.model_max_tokens,
            )

    summary = process_provider(
        provider=args.provider,
        model=model,
        cfg=cfg,
        alias_filter=set(args.alias) if args.alias else None,
        year_min=args.year_min,
        year_max=args.year_max,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    log.info(
        "done: processed=%d skipped=%d total=%d",
        summary["processed"],
        summary["skipped"],
        len(summary["files"]),
    )
    print(
        f"done: processed={summary['processed']} "
        f"skipped={summary['skipped']} total={len(summary['files'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
