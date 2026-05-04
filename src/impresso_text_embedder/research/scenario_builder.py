"""Build the scenario list for one study from a :class:`ScenariosConfig`.

Replaces the hardcoded ``_SCENARIOS`` tuple that used to live in
:mod:`impresso_text_embedder.research.scenarios`. The grid is the
cartesian product ``chunkers × chunk_sizes`` plus an optional
truncate baseline at id ``S0``; ids are auto-numbered ``S0..SN`` in
declaration order so a study cannot have two scenarios colliding on
``(chunker, chunk_tokens)``.

Validation against the chunking + aggregation registries lives here,
not in :mod:`study_config`, so that module stays free of imports
from :mod:`impresso_text_embedder.chunking` and
:mod:`impresso_text_embedder.aggregation`.

Module entry point ``python -m
impresso_text_embedder.research.scenario_builder --config <path>
{--list-ids,--list-table}`` is what the Makefile shells out to so
``SCENARIOS`` stays in sync with the loaded study config without
re-implementing the registry expansion in shell.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from impresso_text_embedder.aggregation import available_strategies as available_aggregators
from impresso_text_embedder.chunking import available_strategies as available_chunkers
from impresso_text_embedder.research.scenarios import Scenario
from impresso_text_embedder.research.study_config import (
    ScenariosConfig,
    StudyConfig,
    load_study_config,
)

_TRUNCATE_LABEL_TOKENS: int = 8192


def build_scenarios(cfg: ScenariosConfig) -> list[Scenario]:
    """Expand ``cfg`` into an ordered list of :class:`Scenario` rows.

    Order: optional truncate baseline first (id ``S0``), then
    ``chunkers × chunk_sizes × aggregators`` in declaration order with
    aggregators as the **innermost** loop. When a study uses a single
    aggregator (the singular ``aggregator:`` form), output is
    bit-identical to the pre-multi-aggregator behaviour: same scenario
    IDs, same labels — so A-fit / B-overflow / v1 keep their
    ``config_sha``. When multiple aggregators are listed, the label
    grows the ``-{agg}`` suffix and "all aggregators at one (chunker,
    size) cell" is a contiguous slice of IDs, which matters for
    smoke-testing the smallest cell first.

    Chunker and aggregator names are checked against the live
    registries; an unknown name aborts at config-build time, before
    any GPU is touched.
    """
    known_chunkers = set(available_chunkers())
    known_aggregators = set(available_aggregators())

    bad_chunkers = [c for c in cfg.chunkers if c not in known_chunkers]
    if bad_chunkers:
        raise ValueError(
            f"unknown chunker(s) {bad_chunkers}; "
            f"registered: {sorted(known_chunkers)}"
        )
    aggs = cfg.effective_aggregators()
    bad_aggs = [a for a in aggs if a not in known_aggregators]
    if bad_aggs:
        raise ValueError(
            f"unknown aggregator(s) {bad_aggs}; "
            f"registered: {sorted(known_aggregators)}"
        )

    label_with_agg = len(aggs) > 1
    out: list[Scenario] = []
    sid = 0
    if cfg.truncate_baseline:
        out.append(
            Scenario(
                id=f"S{sid}",
                label=f"truncate-{_TRUNCATE_LABEL_TOKENS}",
                chunker_name=None,
                chunk_tokens=None,
                aggregator_name=None,
            )
        )
        sid += 1
    for chunker in cfg.chunkers:
        for size in cfg.chunk_sizes:
            for agg in aggs:
                label = (
                    f"{chunker}-{size}-{agg}" if label_with_agg else f"{chunker}-{size}"
                )
                out.append(
                    Scenario(
                        id=f"S{sid}",
                        label=label,
                        chunker_name=chunker,
                        chunk_tokens=size,
                        aggregator_name=agg,
                    )
                )
                sid += 1
    return out


class ScenarioRegistry:
    """Lookup helper around a built scenario list."""

    def __init__(self, scenarios: Sequence[Scenario]) -> None:
        self._scenarios: list[Scenario] = list(scenarios)
        self._by_id: dict[str, Scenario] = {s.id: s for s in self._scenarios}
        if len(self._by_id) != len(self._scenarios):
            raise ValueError(
                "scenario list contains duplicate ids — should not happen "
                "if built via build_scenarios"
            )

    @classmethod
    def from_study(cls, cfg: StudyConfig) -> "ScenarioRegistry":
        return cls(build_scenarios(cfg.scenarios))

    def all_scenarios(self) -> list[Scenario]:
        return list(self._scenarios)

    def all_ids(self) -> list[str]:
        return [s.id for s in self._scenarios]

    def get(self, sid: str) -> Scenario:
        try:
            return self._by_id[sid]
        except KeyError:
            ids = ", ".join(self._by_id)
            raise KeyError(f"unknown scenario {sid!r}; known: {ids}") from None

    def format_table(self) -> str:
        header = f"{'id':<5} {'chunker':<14} {'chunk_tokens':>12}  label"
        sep = "-" * len(header)
        rows = [header, sep]
        for s in self._scenarios:
            chunker = s.chunker_name or "(none)"
            tokens = "-" if s.chunk_tokens is None else str(s.chunk_tokens)
            rows.append(f"{s.id:<5} {chunker:<14} {tokens:>12}  {s.label}")
        return "\n".join(rows)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m impresso_text_embedder.research.scenario_builder",
        description=(
            "Build and inspect the scenario registry for one study. The "
            "Makefile shells out to this to derive the SCENARIOS list at "
            "evaluation time, keeping the build matrix in sync with the "
            "loaded study config."
        ),
    )
    p.add_argument("--config", required=True, type=Path, help="study YAML config path")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--list-ids",
        action="store_true",
        help="space-separated scenario ids on stdout (Makefile-friendly)",
    )
    grp.add_argument(
        "--list-table",
        action="store_true",
        help="formatted table on stdout",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry = ScenarioRegistry.from_study(load_study_config(args.config))
    if args.list_ids:
        print(" ".join(registry.all_ids()))
    else:
        print(registry.format_table())
    return 0


__all__ = ["ScenarioRegistry", "build_scenarios", "main"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
