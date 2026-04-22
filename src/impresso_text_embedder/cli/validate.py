"""Entry point for ``impresso-embed-validate``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from impresso_text_embedder.validate import (
    DEFAULT_TOL,
    ValidationReport,
    validate_against_target,
    validate_structural,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-embed-validate",
        description=(
            "Validate an embedding .jsonl.bz2 file. Without --target: structural "
            "checks only. With --target: per-record cosine distance comparison."
        ),
    )
    p.add_argument("path", help="path or s3:// URI of the produced .jsonl.bz2")
    p.add_argument("--target", default=None, help="reference .jsonl.bz2 to compare against")
    p.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"cosine-distance tolerance (default: {DEFAULT_TOL:g})",
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
    except ImportError:  # pragma: no cover
        return
    load_dotenv()


def _print_report(report: ValidationReport, *, comparing: bool) -> None:
    print(
        f"records_checked={report.records_checked} "
        f"items_checked={report.items_checked} "
        f"max_distance={report.max_distance:.3e}",
    )
    for err in report.errors:
        print(f"  ERROR: {err}")
    limit = 20
    for mm in report.mismatches[:limit]:
        print(f"  MISMATCH: {mm}")
    remaining = len(report.mismatches) - limit
    if remaining > 0:
        print(f"  ... and {remaining} more mismatches")
    if report.passed:
        print("OK: all records within tol" if comparing else "OK")
    else:
        print("FAIL")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_env()

    if args.target:
        report = validate_against_target(args.path, args.target, tol=args.tol)
        _print_report(report, comparing=True)
    else:
        report = validate_structural(args.path)
        _print_report(report, comparing=False)

    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
