"""``Scenario`` dataclass — one cell of the chunking-strategy sweep grid.

The dataclass is the data model only; the registry that used to live
here (``_SCENARIOS``, ``get_scenario``, ``all_scenario_ids``,
``format_table``) moved to :mod:`scenario_builder` along with
``build_scenarios`` and :class:`ScenarioRegistry`. The split keeps
the data shape decoupled from the registry-build logic so the
schema layer (:mod:`study_config`) can hold a forward reference to
``Scenario`` without depending on the chunking + aggregation
registries.

A ``Scenario`` is a named tuple of ``(chunker, chunk_tokens,
aggregator)`` that the embedding-sweep CLI dispatches against the
corpus shard. ``chunker_name`` ``None`` selects the truncate
baseline (no :class:`embed.LongDocConfig` attached; tokenizer drops
the tail at ``model.max_seq_length``). For chunked rows
``chunk_tokens`` doubles as both the trigger threshold
(:attr:`embed.LongDocConfig.model_max_tokens`) and the chunker's
nominal target size — setting them equal forces every doc longer
than the target into the chunker, which is the behaviour the
research question requires.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One row of the sweep grid."""

    id: str
    label: str
    chunker_name: str | None
    chunk_tokens: int | None
    aggregator_name: str | None

    def output_dir_name(self) -> str:
        """Filesystem/S3-key segment: ``S3_fixed-window-1024``."""
        return f"{self.id}_{self.label}"


__all__ = ["Scenario"]
