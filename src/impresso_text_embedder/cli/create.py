"""Entry point for ``impresso-embed-create``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from impresso_text_embedder.embed import EncoderConfig
from impresso_text_embedder.pipeline import PipelineConfig, process_provider


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-embed-create",
        description=(
            "Embed .jsonl.bz2 content-item shards under a given provider on S3, "
            "writing one output .jsonl.bz2 per input file."
        ),
    )
    p.add_argument("--provider", required=True, help="provider code (e.g. SNL)")
    p.add_argument("--input-bucket", required=True)
    p.add_argument("--output-bucket", required=True)
    p.add_argument("--input-prefix", default="", help="optional prefix inside the input bucket")

    p.add_argument("--model-name", default="Alibaba-NLP/gte-multilingual-base")
    p.add_argument("--model-revision", default=None)

    p.add_argument(
        "--embedding-level",
        choices=["text", "sentence", "chunk"],
        default="text",
    )
    p.add_argument(
        "--chunking-strategy",
        default="semantic",
        help="only used when --embedding-level=chunk",
    )

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-char-length", type=int, default=400)
    p.add_argument(
        "--content-type",
        nargs="+",
        default=["ar"],
        choices=["ar", "page"],
    )
    p.add_argument(
        "--normalize-embeddings",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--include-text",
        action="store_true",
        default=False,
        help="only meaningful for --embedding-level=text",
    )

    p.add_argument("--alias", nargs="*", default=None, help="filter to these aliases")
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)

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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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
        normalize_embeddings=args.normalize_embeddings,
        min_char_length=args.min_char_length,
        include_text=args.include_text,
        content_types=frozenset(args.content_type),
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
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_env()

    cfg = _build_pipeline_config(args)

    if args.dry_run:
        model = None  # not used
    else:
        from impresso_text_embedder.model import load_model

        model = load_model(name=args.model_name, revision=args.model_revision)

    summary = process_provider(
        provider=args.provider,
        model=model,
        cfg=cfg,
        alias_filter=set(args.alias) if args.alias else None,
        year_min=args.year_min,
        year_max=args.year_max,
        dry_run=args.dry_run,
    )

    logging.info(
        "done: processed=%d skipped=%d total=%d",
        summary["processed"],
        summary["skipped"],
        len(summary["files"]),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
