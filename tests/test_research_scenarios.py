from __future__ import annotations

import pytest

from impresso_text_embedder.research.scenario_builder import (
    ScenarioRegistry,
    build_scenarios,
)
from impresso_text_embedder.research.scenarios import Scenario
from impresso_text_embedder.research.study_config import ScenariosConfig


def _cfg(
    *,
    chunkers: tuple[str, ...] = ("fixed-window", "token-budget", "semantic"),
    chunk_sizes: tuple[int, ...] = (512, 1024, 2048, 4096, 8190),
    truncate_baseline: bool = True,
    aggregator: str = "mean",
    aggregators: tuple[str, ...] | None = None,
) -> ScenariosConfig:
    kwargs: dict = {
        "truncate_baseline": truncate_baseline,
        "chunkers": chunkers,
        "chunk_sizes": chunk_sizes,
    }
    if aggregators is not None:
        kwargs["aggregators"] = aggregators
    else:
        kwargs["aggregator"] = aggregator
    return ScenariosConfig(**kwargs)


def test_scenario_dataclass_output_dir_name() -> None:
    s = Scenario(
        id="S3", label="fixed-window-2048", chunker_name="fixed-window",
        chunk_tokens=2048, aggregator_name="mean",
    )
    assert s.output_dir_name() == "S3_fixed-window-2048"
    assert "/" not in s.output_dir_name()


def test_build_scenarios_v1_grid_matches_legacy_layout() -> None:
    """The 16-row pre-refactor grid is reproduced exactly from the v1 cfg."""
    scenarios = build_scenarios(_cfg())
    ids = [s.id for s in scenarios]
    assert ids == [f"S{i}" for i in range(16)]

    s0 = scenarios[0]
    assert s0.chunker_name is None
    assert s0.chunk_tokens is None
    assert s0.aggregator_name is None
    assert s0.label == "truncate-8192"

    fixed = [s for s in scenarios if s.chunker_name == "fixed-window"]
    token_budget = [s for s in scenarios if s.chunker_name == "token-budget"]
    semantic = [s for s in scenarios if s.chunker_name == "semantic"]
    assert {s.chunk_tokens for s in fixed} == {512, 1024, 2048, 4096, 8190}
    assert {s.chunk_tokens for s in token_budget} == {512, 1024, 2048, 4096, 8190}
    assert {s.chunk_tokens for s in semantic} == {512, 1024, 2048, 4096, 8190}
    for s in fixed + token_budget + semantic:
        assert s.aggregator_name == "mean"


def test_build_scenarios_truncate_baseline_off() -> None:
    scenarios = build_scenarios(_cfg(truncate_baseline=False))
    assert all(s.chunker_name is not None for s in scenarios)
    assert scenarios[0].id == "S0"  # ids start at 0 regardless


def test_build_scenarios_subset_grid() -> None:
    """Study A's grid: 4 sizes × 2 chunkers + truncate = 9 scenarios."""
    scenarios = build_scenarios(_cfg(
        chunkers=("fixed-window", "token-budget"),
        chunk_sizes=(512, 1024, 2048, 4096),
    ))
    assert len(scenarios) == 1 + 2 * 4
    assert {s.chunker_name for s in scenarios} == {None, "fixed-window", "token-budget"}
    assert {s.chunk_tokens for s in scenarios if s.chunker_name == "fixed-window"} == {
        512, 1024, 2048, 4096,
    }


def test_build_scenarios_unknown_chunker_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        build_scenarios(_cfg(chunkers=("not-a-chunker",)))
    assert "not-a-chunker" in str(exc.value)


def test_build_scenarios_unknown_aggregator_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        build_scenarios(_cfg(aggregator="weighted-mean"))
    assert "weighted-mean" in str(exc.value)


def test_chunk_tokens_match_label_for_chunked_scenarios() -> None:
    for s in build_scenarios(_cfg()):
        if s.chunker_name is None:
            continue
        assert str(s.chunk_tokens) in s.label
        assert s.chunker_name in s.label


# ---------------------------------------------------------------------------
# ScenarioRegistry
# ---------------------------------------------------------------------------


def test_registry_get_round_trips() -> None:
    registry = ScenarioRegistry(build_scenarios(_cfg()))
    s = registry.get("S3")
    assert s.id == "S3"
    assert s.chunker_name == "fixed-window"


def test_registry_unknown_id_lists_known() -> None:
    registry = ScenarioRegistry(build_scenarios(_cfg()))
    with pytest.raises(KeyError) as exc:
        registry.get("Sxxx")
    msg = str(exc.value)
    assert "Sxxx" in msg
    assert "S0" in msg


def test_registry_format_table_lists_every_id() -> None:
    registry = ScenarioRegistry(build_scenarios(_cfg()))
    table = registry.format_table()
    for sid in registry.all_ids():
        assert sid in table


def test_registry_rejects_duplicate_ids() -> None:
    s = Scenario(
        id="S0", label="x", chunker_name=None, chunk_tokens=None,
        aggregator_name=None,
    )
    with pytest.raises(ValueError):
        ScenarioRegistry([s, s])


# ---------------------------------------------------------------------------
# Multi-aggregator (plural) cartesian
# ---------------------------------------------------------------------------


def test_singleton_aggregator_label_unchanged_for_backcompat() -> None:
    """Singular aggregator path keeps the legacy ``{chunker}-{size}`` label."""
    scenarios = build_scenarios(_cfg(
        chunkers=("token-budget",), chunk_sizes=(1024,), aggregator="mean",
    ))
    chunked = [s for s in scenarios if s.chunker_name is not None]
    assert chunked[0].label == "token-budget-1024"
    assert chunked[0].aggregator_name == "mean"


def test_multi_aggregator_grid_size_and_order() -> None:
    """Plural aggregators expand cartesian with aggregator innermost."""
    aggs = ("mean", "max", "first-chunk", "length-weighted")
    scenarios = build_scenarios(_cfg(
        chunkers=("token-budget",),
        chunk_sizes=(256, 1024, 4096),
        aggregators=aggs,
    ))
    # 1 baseline + 1 chunker × 3 sizes × 4 aggregators = 13 scenarios.
    assert len(scenarios) == 1 + 1 * 3 * 4
    assert [s.id for s in scenarios] == [f"S{i}" for i in range(13)]
    chunked = [s for s in scenarios if s.chunker_name is not None]
    # Size-major, aggregator innermost: contiguous slice of 4 per size.
    expected_pairs = [
        (size, agg) for size in (256, 1024, 4096) for agg in aggs
    ]
    actual_pairs = [(s.chunk_tokens, s.aggregator_name) for s in chunked]
    assert actual_pairs == expected_pairs


def test_multi_aggregator_label_carries_agg_suffix() -> None:
    aggs = ("mean", "max")
    scenarios = build_scenarios(_cfg(
        chunkers=("token-budget",),
        chunk_sizes=(1024,),
        aggregators=aggs,
    ))
    chunked = [s for s in scenarios if s.chunker_name is not None]
    assert {s.label for s in chunked} == {
        "token-budget-1024-mean",
        "token-budget-1024-max",
    }


def test_multi_aggregator_unknown_name_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        build_scenarios(_cfg(
            chunkers=("token-budget",),
            chunk_sizes=(1024,),
            aggregators=("mean", "not-a-real-aggregator"),
        ))
    assert "not-a-real-aggregator" in str(exc.value)


def test_multi_aggregator_baseline_still_emits_S0() -> None:
    scenarios = build_scenarios(_cfg(
        chunkers=("token-budget",),
        chunk_sizes=(256,),
        aggregators=("mean", "max"),
        truncate_baseline=True,
    ))
    assert scenarios[0].id == "S0"
    assert scenarios[0].chunker_name is None
    assert scenarios[0].aggregator_name is None
    assert scenarios[0].label == "truncate-8192"
