"""Entry point for ``impresso-embed-validate``."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from impresso_text_embedder.csv_export import export_report_to_csv
from impresso_text_embedder.validate import (
    DEFAULT_SOURCE_MIN_CHAR_LENGTH,
    DEFAULT_TOL,
    CharLengthStats,
    Mismatch,
    MismatchKind,
    Sample,
    SourceStatsAnalysis,
    SourceStatsBlock,
    ValidationReport,
    collect_source_stats,
    validate_against_target,
    validate_structural,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-embed-validate",
        description=(
            "Validate an embedding .jsonl.bz2 file. Without --target: structural "
            "checks only. With --target: per-record cosine distance comparison. "
            "Pass --source to cross-reference missing records against the "
            "original input shard."
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
        "--source",
        default=None,
        help=(
            "original input .jsonl.bz2 (local or s3:// URI) — when combined with "
            "--target, computes per-direction statistics on the records that show "
            "up as missing in target / missing in produced"
        ),
    )
    p.add_argument(
        "--source-min-char-length",
        type=int,
        default=DEFAULT_SOURCE_MIN_CHAR_LENGTH,
        help=(
            "character-length threshold used to tally below-min-char records in "
            f"the source-stats block (default: {DEFAULT_SOURCE_MIN_CHAR_LENGTH}, "
            "matches impresso-embed-create's default)"
        ),
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of worst drifts to list (default: 10)",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help=(
            "write above_threshold.csv and missing.csv into this directory "
            "(per-row Impresso article URL is included for click-through). "
            "Source-derived columns (lg, tp, char_length, …) are populated "
            "only when --source is also passed; otherwise blank."
        ),
    )
    p.add_argument(
        "--show-all-missing",
        action="store_true",
        help="list every missing id instead of a head summary",
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


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _log_histogram(distances: list[float], tol: float, width: int = 40) -> list[str]:
    """Log10-binned histogram. Returns lines to print.

    Bins (log10): (-inf,-7], (-7,-6], ..., (-1, 0]. Zero distances land in the
    lowest bin. ``tol`` is flagged with an arrow on whichever bin contains it.
    """
    edges = [-7, -6, -5, -4, -3, -2, -1, 0]  # upper edges in log10 space
    labels = [
        "  <= -7  ",
        " -7 .. -6",
        " -6 .. -5",
        " -5 .. -4",
        " -4 .. -3",
        " -3 .. -2",
        " -2 .. -1",
        " -1 ..  0",
    ]
    counts = [0] * len(edges)
    for d in distances:
        if d <= 0:
            counts[0] += 1
            continue
        lg = math.log10(d)
        placed = False
        for i, ub in enumerate(edges):
            if lg <= ub:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    tol_bin: int | None = None
    if tol > 0:
        ltol = math.log10(tol)
        for i, ub in enumerate(edges):
            if ltol <= ub:
                tol_bin = i
                break

    max_count = max(counts) if counts else 0
    count_w = max(len(str(c)) for c in counts) if counts else 1
    lines = ["  log10(distance)   count"]
    for i, c in enumerate(counts):
        bar = "█" * int(round((c / max_count) * width)) if max_count else ""
        marker = f"   ← tol={tol:.0e}" if i == tol_bin else ""
        lines.append(f"  {labels[i]}   {str(c).rjust(count_w)}  {bar}{marker}")
    return lines


def _length_histogram(lengths: list[int], width: int = 40) -> list[str]:
    """Log10-binned histogram for char lengths. Returns lines to print."""
    edges = [0, 1, 2, 3, 4, 5]  # upper edges in log10 space (1, 10, 100, 1k, 10k, 100k)
    labels = [
        "      0 ..     1",
        "      1 ..    10",
        "     10 ..   100",
        "    100 ..    1k",
        "     1k ..   10k",
        "    10k ..  100k",
        "   100k+        ",
    ]
    counts = [0] * (len(edges) + 1)
    for length in lengths:
        if length <= 0:
            counts[0] += 1
            continue
        lg = math.log10(length)
        placed = False
        for i, ub in enumerate(edges):
            if lg <= ub:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    max_count = max(counts) if counts else 0
    count_w = max(len(str(c)) for c in counts) if counts else 1
    lines = ["  char length       count"]
    for i, c in enumerate(counts):
        bar = "█" * int(round((c / max_count) * width)) if max_count else ""
        lines.append(f"  {labels[i]}    {str(c).rjust(count_w)}  {bar}")
    return lines


def _sorted_value_mismatches(mismatches: list[Mismatch]) -> list[Mismatch]:
    vals = [m for m in mismatches if m.kind == MismatchKind.VALUE and m.distance is not None]
    vals.sort(key=lambda m: m.distance, reverse=True)  # type: ignore[arg-type,return-value]
    return vals


def _render_missing(
    missing: list[Mismatch],
    label: str,
    *,
    show_all: bool,
    head: int = 6,
) -> list[str]:
    if not missing:
        return []
    ids = [m._qualifier().removeprefix("ci_id=").strip("'") for m in missing]
    lines = [f"  {label} ({len(missing)})"]
    if show_all or len(ids) <= head:
        for chunk_start in range(0, len(ids), 4):
            lines.append("    " + ", ".join(ids[chunk_start : chunk_start + 4]))
    else:
        lines.append("    " + ", ".join(ids[:head]) + f", ... (+{len(ids) - head} more)")
    return lines


def _worst_id_text(m: Mismatch) -> str:
    if m.id_key and m.item_id is not None:
        return f"{m.ci_id} {m.id_key}={m.item_id}"
    return m.ci_id


def _counts_table(report: ValidationReport, *, comparing: bool, tol: float) -> Table:
    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column("key", style="cyan", no_wrap=True)
    t.add_column("value")
    t.add_row("records checked", str(report.records_checked))
    t.add_row("items checked", str(report.items_checked))
    if report.level is not None:
        t.add_row("level", report.level)
    if comparing:
        t.add_row("tolerance", f"{tol:.3e}")
        t.add_row("max distance", f"{report.max_distance:.3e}")
    return t


def _breakdown_table(
    *,
    value_count: int,
    miss_target_count: int,
    miss_produced_count: int,
    compared: int,
) -> Table:
    total = max(compared + miss_target_count + miss_produced_count, 1)

    def pct(n: int) -> str:
        return f"{n / total * 100:5.1f}%"

    t = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    t.add_column("kind", no_wrap=True)
    t.add_column("count", justify="right")
    t.add_column("pct", justify="right")
    passing = compared - value_count
    t.add_row("above tolerance", str(value_count), pct(value_count))
    t.add_row("missing in target", str(miss_target_count), pct(miss_target_count))
    t.add_row("missing in produced", str(miss_produced_count), pct(miss_produced_count))
    t.add_row("passing", str(passing), pct(passing))
    return t


def _worst_drifts_table(shown: list[Mismatch]) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    t.add_column("ci_id", no_wrap=True)
    t.add_column("cosine_distance", justify="right")
    for m in shown:
        t.add_row(_worst_id_text(m), f"{m.distance:.3e}")
    return t


# --------------------------------------------------------------------------- #
# Source-stats rendering
# --------------------------------------------------------------------------- #


_DIRECTION_LABELS: dict[MismatchKind, str] = {
    MismatchKind.VALUE: "above tolerance",
    MismatchKind.MISSING_IN_TARGET: "missing in target",
    MismatchKind.MISSING_IN_PRODUCED: "missing in produced",
}


_DIRECTION_BORDER: dict[MismatchKind, str] = {
    MismatchKind.VALUE: "red",
    MismatchKind.MISSING_IN_TARGET: "yellow",
    MismatchKind.MISSING_IN_PRODUCED: "yellow",
}


def _format_sample_line(sample: Sample) -> str:
    if sample.distance is not None:
        return f"  {sample.ci_id}  d={sample.distance:.2e}  {sample.excerpt!r}"
    return f"  {sample.ci_id}  {sample.excerpt!r}"


def _source_block_counts_table(block: SourceStatsBlock, min_char_length: int) -> Table:
    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column("key", style="cyan", no_wrap=True)
    t.add_column("value", justify="right")
    t.add_row("total", str(block.total))
    t.add_row("found in source", str(block.found_in_source))
    t.add_row("not in source", str(len(block.not_in_source_ids)))
    t.add_row("reconstructable", str(block.reconstructable))
    t.add_row("empty (no ft/sents)", str(block.empty))
    t.add_row(f"below min_char {min_char_length}", str(block.below_min_char))
    return t


def _source_block_distribution_table(
    counts,
    *,
    title: str,
    key_label: str,
) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, pad_edge=False, title=title, title_justify="left")
    t.add_column(key_label, no_wrap=True)
    t.add_column("count", justify="right")
    for code, cnt in counts.most_common():
        t.add_row(code, str(cnt))
    return t


def _format_baseline_line(baseline: CharLengthStats) -> str | None:
    if baseline.count == 0 or baseline.mean is None:
        return None
    return (
        f"  vs kept (n={baseline.count})  min={baseline.min}  "
        f"mean={baseline.mean:.0f}  max={baseline.max}"
    )


def _render_source_stats_block(
    block: SourceStatsBlock,
    min_char_length: int,
    baseline: CharLengthStats | None = None,
) -> Panel:
    label = _DIRECTION_LABELS[block.direction]
    title = f"source analysis — {label} ({block.total} records)"

    parts = [_source_block_counts_table(block, min_char_length)]

    if block.char_lengths:
        sd = sorted(block.char_lengths)
        mean = sum(sd) / len(sd)
        stats_line = (
            f"char length (this set, n={len(sd)})  min={sd[0]}  "
            f"p50={int(_percentile(sd, 0.5))}  p90={int(_percentile(sd, 0.9))}  "
            f"p99={int(_percentile(sd, 0.99))}  mean={mean:.0f}  max={sd[-1]}"
        )
        parts.append(Text(stats_line))
        if baseline is not None:
            line = _format_baseline_line(baseline)
            if line is not None:
                parts.append(Text(line))
        parts.append(Text("\n".join(_length_histogram(sd))))

    if block.lg_counts:
        parts.append(
            _source_block_distribution_table(
                block.lg_counts, title="language", key_label="lg"
            )
        )
    if block.tp_counts:
        parts.append(
            _source_block_distribution_table(
                block.tp_counts, title="content type", key_label="tp"
            )
        )

    if block.samples:
        sample_lines = ["samples"] + [_format_sample_line(s) for s in block.samples]
        parts.append(Text("\n".join(sample_lines)))

    border_style = _DIRECTION_BORDER.get(block.direction, "yellow")
    return Panel(Group(*parts), title=title, title_align="left", border_style=border_style)


def _emit_source_stats(console: Console, analysis: SourceStatsAnalysis | None) -> None:
    if analysis is None:
        return
    # VALUE (drifted) first so a reader scanning top-to-bottom sees
    # "what drifted" before "what's missing".
    for direction in (
        MismatchKind.VALUE,
        MismatchKind.MISSING_IN_TARGET,
        MismatchKind.MISSING_IN_PRODUCED,
    ):
        block = analysis.blocks.get(direction)
        if block is None or block.total == 0:
            continue
        console.print()
        console.print(
            _render_source_stats_block(
                block, analysis.min_char_length, baseline=analysis.baseline
            )
        )


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def _print_report(
    report: ValidationReport,
    *,
    comparing: bool,
    tol: float,
    top: int,
    show_all_missing: bool,
    path: str,
    target: str | None,
    console: Console | None = None,
) -> None:
    console = console or Console()

    # (a) Header
    header = f"{path}" + (f" → {target}" if target else "")
    console.print(header)

    # Errors always first.
    for err in report.errors:
        console.print(f"  ERROR: {err}")

    # Legacy single-line summary (kept for test substrings and quick grepping).
    console.print(
        f"records_checked={report.records_checked} "
        f"items_checked={report.items_checked} "
        f"max_distance={report.max_distance:.3e}"
    )
    # (b) Counts table
    console.print(_counts_table(report, comparing=comparing, tol=tol))

    if comparing:
        value_mms = [m for m in report.mismatches if m.kind == MismatchKind.VALUE]
        miss_target = [m for m in report.mismatches if m.kind == MismatchKind.MISSING_IN_TARGET]
        miss_produced = [m for m in report.mismatches if m.kind == MismatchKind.MISSING_IN_PRODUCED]
        compared = len(report.distances)

        console.print()
        console.print("  mismatches")
        console.print(
            _breakdown_table(
                value_count=len(value_mms),
                miss_target_count=len(miss_target),
                miss_produced_count=len(miss_produced),
                compared=compared,
            )
        )

        # (d) Distance percentiles
        if report.distances:
            sd = sorted(report.distances)
            stats = (
                f"min={sd[0]:.3e}  p50={_percentile(sd, 0.5):.3e}  "
                f"p90={_percentile(sd, 0.9):.3e}  p99={_percentile(sd, 0.99):.3e}  "
                f"max={sd[-1]:.3e}"
            )
            over = len(value_mms)
            pct = over / compared * 100 if compared else 0.0
            console.print()
            console.print("  distance")
            console.print(f"    {stats}")
            console.print(f"    > tol ({tol:.0e})  count={over} / {compared}  ({pct:.1f}%)")

            # (e) Histogram
            console.print()
            for line in _log_histogram(report.distances, tol):
                console.print(line)

        # (f) Top-N worst drifts
        worst = _sorted_value_mismatches(report.mismatches)
        if worst:
            shown = worst[:top]
            console.print()
            console.print(f"  worst drifts (top {len(shown)} of {len(worst)})")
            console.print(_worst_drifts_table(shown))
            # One legacy MISMATCH: line keeps grep contracts alive without echoing the table.
            console.print(f"  MISMATCH: {shown[0]}")

        # (g) Missing summaries
        if miss_target or miss_produced:
            console.print()
            for line in _render_missing(
                miss_target, "missing in target", show_all=show_all_missing
            ):
                console.print(line)
            for line in _render_missing(
                miss_produced, "missing in produced", show_all=show_all_missing
            ):
                console.print(line)
            # Legacy MISMATCH: marker so substring tests still see missing cases.
            if miss_target:
                console.print(f"  MISMATCH: {miss_target[0]}")
            if miss_produced:
                console.print(f"  MISMATCH: {miss_produced[0]}")

        # (g') Source-stats panels — only when data was gathered and that
        # direction has mismatches.
        _emit_source_stats(console, report.source_stats)

    # (h) Verdict
    console.print()
    if report.passed:
        console.print("OK: all records within tol" if comparing else "OK")
    else:
        console.print("FAIL")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_env()

    console = Console()

    if args.target:
        report = validate_against_target(args.path, args.target, tol=args.tol)
        if args.source:
            try:
                collect_source_stats(
                    args.source,
                    report,
                    min_char_length=args.source_min_char_length,
                )
            except (OSError, ValueError) as exc:
                print(f"WARNING: --source failed: {exc}", file=sys.stderr)
        _print_report(
            report,
            comparing=True,
            tol=args.tol,
            top=args.top,
            show_all_missing=args.show_all_missing,
            path=args.path,
            target=args.target,
            console=console,
        )
    else:
        if args.source:
            print(
                "WARNING: --source ignored without --target "
                "(structural mode has no missing-records analysis)",
                file=sys.stderr,
            )
        report = validate_structural(args.path)
        _print_report(
            report,
            comparing=False,
            tol=args.tol,
            top=args.top,
            show_all_missing=args.show_all_missing,
            path=args.path,
            target=None,
            console=console,
        )

    if args.csv_out is not None:
        above_path, missing_path = export_report_to_csv(report, args.csv_out)
        print(f"wrote {above_path}")
        print(f"wrote {missing_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
