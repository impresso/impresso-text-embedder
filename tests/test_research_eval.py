"""Tests for :mod:`impresso_text_embedder.research.eval`.

Loaders are exercised via a tmp-path round-trip (write
``.jsonl.bz2`` shards on disk, point a study config at them via
monkeypatched ``ensure_local``); metrics are exercised on small
deterministic numpy fixtures with known answers.
"""

from __future__ import annotations

import bz2
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pytest

from impresso_text_embedder.research import eval as ev
from impresso_text_embedder.research.scenarios import Scenario
from impresso_text_embedder.research.study_config import (
    QUERIES_EMBEDDED_FILENAME,
    StudyConfig,
    scenario_filename,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 0 else arr.tolist()


def _write_jsonl_bz2(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wb") as fh:
        for r in records:
            fh.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))


def _make_study_config(tmp_path: Path) -> StudyConfig:
    """Build a StudyConfig pointing at tmp_path (no S3, no real corpus)."""
    return StudyConfig.model_validate(
        {
            "study": {"name": "T1"},
            "s3": {"bucket": "test-bucket", "rebuilt_bucket": "rebuilt"},
            "paths": {
                "local_root": str(tmp_path / "studies/{study}"),
                "s3_root": "chunking-eval/{study}",
            },
            "corpus": {
                "input_path": "/dev/null",
                "languages": ["fr", "de"],
                "ocrqa_min": 0.9,
                "year_min": 1900,
                "year_max": 1950,
                "providers": {"fr": ["BNF"], "de": ["SNL"]},
                "chars_per_token": {"fr": 4.5, "de": 3.5},
                "n_per_lg": 4,
                "min_tokens": 1024,
                "max_tokens": 8192,
            },
            "embed": {
                "model_name": "Alibaba-NLP/gte-multilingual-base",
                "model_revision": "f7d567e",
            },
            "scenarios": {
                "truncate_baseline": True,
                "chunkers": ["fixed-window"],
                "chunk_sizes": [512, 1024],
                "aggregator": "mean",
            },
            "query_generation": {
                "endpoint": "http://x",
                "model": "M",
                "max_parallel": 1,
                "temperature": 0.0,
                "max_output_tokens": 128,
                "request_timeout_s": 30.0,
                "retry_attempts": 1,
                "position_buckets": ["head", "mid", "tail"],
            },
        }
    )


def _split_tokens(total: int, n: int) -> list[int]:
    """Split ``total`` into ``n`` near-equal positive ints summing to ``total``."""
    base = total // n
    rem = total - base * n
    return [base + 1 if i < rem else base for i in range(n)]


# Per-doc fixture totals chosen so each lands in a distinct token_bucket
# under DEFAULT_TOKEN_BUCKET_EDGES = (2048, 4096, 6144, 8192):
#   fr-1: 3500 → "2k-4k"
#   fr-2: 4500 → "4k-6k"
#   de-1: 7000 → "6k-8k"
#   de-2: 9100 → "8k+"  (also exceeds the 8190 truncation limit on S0)
_FIXTURE_DOC_TOKENS: dict[str, int] = {
    "fr-1": 3500,
    "fr-2": 4500,
    "de-1": 7000,
    "de-2": 9100,
}


def _per_chunk_tokens(ci_id: str, scenario: Scenario, n_chunks: int) -> list[int]:
    total = _FIXTURE_DOC_TOKENS[ci_id]
    return [total] if scenario.id == "S0" else _split_tokens(total, n_chunks)


def _seed_pool(study_cfg: StudyConfig, scenario: Scenario, *, dim: int = 4) -> dict:
    """Write a 4-doc per-scenario shard (2 fr + 2 de) with deterministic
    near-orthogonal embeddings; return the raw record list for assertions."""
    # Each doc points at a different basis vector so cosine is sharp.
    bases = np.eye(dim, dtype=np.float32)
    fr_n_chunks = 1 if scenario.id == "S0" else 4
    de_n_chunks = 1 if scenario.id == "S0" else 8
    records = [
        {
            "ci_id": "fr-1",
            "lg": "fr",
            "year": 1900,
            "provider": "BNF",
            "alias": "le-temps",
            "ocrqa": 0.95,
            "len_chars": 30000,
            "n_chunks": fr_n_chunks,
            "n_tokens": _FIXTURE_DOC_TOKENS["fr-1"],
            "n_tokens_per_chunk": _per_chunk_tokens("fr-1", scenario, fr_n_chunks),
            "scenario_id": scenario.id,
            "chunker": scenario.chunker_name or "truncate",
            "chunk_tokens": scenario.chunk_tokens,
            "embedding": _unit(bases[0].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "ci_type": "ar",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
        {
            "ci_id": "fr-2",
            "lg": "fr",
            "year": 1910,
            "provider": "BNF",
            "alias": "le-temps",
            "ocrqa": 0.95,
            "len_chars": 32000,
            "n_chunks": fr_n_chunks,
            "n_tokens": _FIXTURE_DOC_TOKENS["fr-2"],
            "n_tokens_per_chunk": _per_chunk_tokens("fr-2", scenario, fr_n_chunks),
            "scenario_id": scenario.id,
            "chunker": scenario.chunker_name or "truncate",
            "chunk_tokens": scenario.chunk_tokens,
            "embedding": _unit(bases[1].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "ci_type": "ar",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
        {
            "ci_id": "de-1",
            "lg": "de",
            "year": 1900,
            "provider": "SNL",
            "alias": "snl",
            "ocrqa": 0.95,
            "len_chars": 28000,
            "n_chunks": de_n_chunks,
            "n_tokens": _FIXTURE_DOC_TOKENS["de-1"],
            "n_tokens_per_chunk": _per_chunk_tokens("de-1", scenario, de_n_chunks),
            "scenario_id": scenario.id,
            "chunker": scenario.chunker_name or "truncate",
            "chunk_tokens": scenario.chunk_tokens,
            "embedding": _unit(bases[2].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "ci_type": "ar",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
        {
            "ci_id": "de-2",
            "lg": "de",
            "year": 1920,
            "provider": "SNL",
            "alias": "snl",
            "ocrqa": 0.95,
            "len_chars": 27000,
            "n_chunks": de_n_chunks,
            "n_tokens": _FIXTURE_DOC_TOKENS["de-2"],
            "n_tokens_per_chunk": _per_chunk_tokens("de-2", scenario, de_n_chunks),
            "scenario_id": scenario.id,
            "chunker": scenario.chunker_name or "truncate",
            "chunk_tokens": scenario.chunk_tokens,
            "embedding": _unit(bases[3].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "ci_type": "ar",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
    ]
    _write_jsonl_bz2(study_cfg.local_path(scenario_filename(scenario.id)), records)
    return records


def _seed_queries(study_cfg: StudyConfig, *, dim: int = 4) -> list[dict]:
    """Two fr queries, two de queries, each near-aligned with their gold doc."""
    bases = np.eye(dim, dtype=np.float32)
    records = [
        {
            "query_id": "fr-1__head__00",
            "ci_id": "fr-1",
            "lg": "fr",
            "query_type": "question",
            "position_bucket": "head",
            "position_chars": [0, 10000],
            "references": [{"text": "verbatim", "char_start": 100, "char_end": 110}],
            "query_text": "Quelle est la question ?",
            "embedding": _unit(bases[0].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
        {
            "query_id": "fr-2__tail__00",
            "ci_id": "fr-2",
            "lg": "fr",
            "query_type": "topical-phrase",
            "position_bucket": "tail",
            "position_chars": [20000, 32000],
            "references": [],
            "query_text": "phrase topique",
            "embedding": _unit(bases[1].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
        {
            "query_id": "de-1__mid__00",
            "ci_id": "de-1",
            "lg": "de",
            "query_type": "question",
            "position_bucket": "mid",
            "position_chars": [9000, 18000],
            "references": [],
            "query_text": "Wie heisst es?",
            "embedding": _unit(bases[2].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
        {
            "query_id": "de-2__head__00",
            "ci_id": "de-2",
            "lg": "de",
            "query_type": "topical-phrase",
            "position_bucket": "head",
            "position_chars": [0, 9000],
            "references": [],
            "query_text": "thematische Phrase",
            "embedding": _unit(bases[3].tolist()),
            "size": dim,
            "model_id": "test@x",
            "ts": "2026-01-01T00:00:00Z",
            "study_name": study_cfg.study.name,
            "study_config_sha": study_cfg.config_sha,
        },
    ]
    _write_jsonl_bz2(study_cfg.local_path(QUERIES_EMBEDDED_FILENAME), records)
    return records


@pytest.fixture
def seeded_study(tmp_path, monkeypatch):
    """Pre-write all eval inputs to local mirror; bypass the S3 download."""
    cfg = _make_study_config(tmp_path)
    scenarios = [
        Scenario(id="S0", label="truncate-8192", chunker_name=None, chunk_tokens=None, aggregator_name=None),
        Scenario(id="S1", label="fixed-window-512", chunker_name="fixed-window", chunk_tokens=512, aggregator_name="mean"),
        Scenario(id="S2", label="fixed-window-1024", chunker_name="fixed-window", chunk_tokens=1024, aggregator_name="mean"),
    ]
    for scen in scenarios:
        _seed_pool(cfg, scen)
    _seed_queries(cfg)

    # Bypass S3: ensure_local just returns the local mirror path.
    def _fake_ensure_local(study_cfg, filename, *, force=False):
        return study_cfg.local_path(filename)

    monkeypatch.setattr(ev, "ensure_local", _fake_ensure_local)
    return cfg, scenarios


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_load_queries_round_trip(seeded_study):
    cfg, _ = seeded_study
    queries = ev.load_queries(cfg)
    assert len(queries) == 4
    fr_q = next(q for q in queries if q.query_id == "fr-1__head__00")
    assert fr_q.lg == "fr"
    assert fr_q.position_bucket == "head"
    assert fr_q.position_chars == (0, 10000)
    assert fr_q.embedding.shape == (4,)
    assert pytest.approx(np.linalg.norm(fr_q.embedding), rel=1e-5) == 1.0


def test_load_scenario_pool_round_trip(seeded_study):
    cfg, scenarios = seeded_study
    pool = ev.load_scenario_pool(cfg, scenarios[1])
    assert pool.embeddings.shape == (4, 4)
    assert sorted(pool.ci_ids.tolist()) == ["de-1", "de-2", "fr-1", "fr-2"]
    fr_mask = pool.lang_mask("fr")
    assert int(fr_mask.sum()) == 2
    assert pool.index_of("fr-1") is not None
    assert pool.index_of("missing") is None


def test_load_eval_inputs_full_registry(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    assert len(inputs.queries) == 4
    assert set(inputs.pools.keys()) == {"S0", "S1", "S2"}


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def test_sanity_clean_inputs_pass(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    report = ev.run_sanity(inputs)
    assert report.passed, report.issues
    assert report.embedding_dim == 4
    assert report.queries_lg_counts == {"fr": 2, "de": 2}
    assert report.pool_size_per_scenario == {"S0": 4, "S1": 4, "S2": 4}
    assert report.queries_with_unmatched_ci_id == []


def test_sanity_flags_unmatched_ci_id(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    bad_query = ev.QueryRow(
        query_id="ghost__head__00",
        ci_id="ghost-doc",
        lg="fr",
        query_type="question",
        position_bucket="head",
        position_chars=(0, 100),
        references=(),
        query_text="?",
        embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    queries = inputs.queries + (bad_query,)
    report = ev.run_sanity(
        ev.EvalInputs(
            study=inputs.study,
            queries=queries,
            pools=inputs.pools,
            scenarios=inputs.scenarios,
        )
    )
    assert not report.passed
    assert "ghost__head__00" in report.queries_with_unmatched_ci_id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_cosine_scores_unit_vectors():
    pool = np.eye(4, dtype=np.float32)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scores = ev.cosine_scores(q, pool)
    np.testing.assert_allclose(scores, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_cosine_scores_handles_zero_vectors():
    pool = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    q = np.array([1.0, 0.0], dtype=np.float32)
    scores = ev.cosine_scores(q, pool)
    assert np.isfinite(scores).all()


def test_rank_of_gold_breaks_ties_pessimistically():
    # Three pool entries: 1 has higher score, 1 ties with gold, gold itself.
    scores = np.array([0.9, 0.5, 0.5], dtype=np.float32)
    # gold is index 2, score 0.5; tied with index 1, beaten by index 0.
    # Pessimistic: tied items count as "higher or equal" → rank 3.
    assert ev.rank_of_gold(scores, gold_idx=2) == 3


def test_rank_of_gold_strict_winner():
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    assert ev.rank_of_gold(scores, gold_idx=1) == 1


def test_recall_at_k_basic():
    ranks = [1, 2, 5, 10, 50]
    assert ev.recall_at_k(ranks, 1) == pytest.approx(0.2)
    assert ev.recall_at_k(ranks, 5) == pytest.approx(0.6)
    assert ev.recall_at_k(ranks, 10) == pytest.approx(0.8)


def test_mean_reciprocal_rank():
    ranks = [1, 2, 4]  # 1/1 + 1/2 + 1/4 = 1.75 / 3
    assert ev.mean_reciprocal_rank(ranks) == pytest.approx(1.75 / 3)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.6, scale=0.1, size=200)
    point, low, high = ev.bootstrap_ci(values, n_iter=500, seed=0)
    assert low < point < high
    assert abs(point - float(np.mean(values))) < 1e-9
    # CI width should be small with n=200, so bound it loosely:
    assert (high - low) < 0.1


def test_bootstrap_ci_handles_empty():
    point, low, high = ev.bootstrap_ci([])
    assert all(np.isnan(x) for x in (point, low, high))


def test_paired_bootstrap_delta_detects_uplift():
    rng = np.random.default_rng(1)
    n = 300
    base = rng.binomial(1, 0.5, size=n).astype(float)
    # 10pp uplift: flip a deterministic subset of zeros to ones.
    boosted = base.copy()
    zeros = np.where(base == 0)[0][:30]
    boosted[zeros] = 1
    delta, low, high = ev.paired_bootstrap_delta(boosted, base, n_iter=500, seed=0)
    assert delta > 0
    assert low > 0  # CI excludes zero — uplift is significant


def test_paired_bootstrap_delta_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ev.paired_bootstrap_delta([1.0, 0.0], [1.0])


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------


def test_score_queries_returns_tidy_dataframe(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    # 4 queries × 3 scenarios = 12 rows.
    assert len(df) == 12
    expected_cols = {
        "query_id",
        "ci_id",
        "lg",
        "query_type",
        "position_bucket",
        "scenario_id",
        "scenario_label",
        "chunker",
        "chunk_tokens",
        "pool_size",
        "rank",
        "reciprocal_rank",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "n_chunks",
    }
    assert expected_cols.issubset(df.columns)
    # Per-language pool: fr queries see 2 fr docs, de queries see 2 de docs.
    fr_rows = df[df.lg == "fr"]
    assert (fr_rows.pool_size == 2).all()


def test_score_queries_ranks_perfect_alignment_first(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    # Embeddings are basis-aligned with their gold doc — every query
    # should rank its gold ci_id at rank 1 in every scenario.
    assert (df["rank"] == 1).all()
    assert (df["recall_at_1"]).all()
    assert df["reciprocal_rank"].astype(float).eq(1.0).all()


def test_aggregate_recall_with_ci_one_row_per_cell(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    agg = ev.aggregate_recall_with_ci(df, metric="recall_at_5", n_iter=200)
    # 2 langs × 3 scenarios = 6 cells.
    assert len(agg) == 6
    # All recall is 1.0 by construction (perfect alignment).
    assert agg["point"].astype(float).eq(1.0).all()


def test_baseline_delta_table_excludes_baseline_row(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    table = ev.baseline_delta_table(df, metric="recall_at_5", baseline_id="S0", n_iter=200)
    assert "S0" not in table["scenario_id"].tolist()
    # All deltas are zero by construction.
    assert pd.Series(table["delta"]).abs().max() == pytest.approx(0.0)


def test_baseline_delta_table_unknown_baseline_raises(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    with pytest.raises(ValueError):
        ev.baseline_delta_table(df, baseline_id="S99")


# ---------------------------------------------------------------------------
# Token-bucket helpers + per-doc token metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "unknown"),
        (0, "unknown"),
        (1, "<2k"),
        (2047, "<2k"),
        (2048, "2k-4k"),
        (4095, "2k-4k"),
        (4096, "4k-6k"),
        (6144, "6k-8k"),
        (8191, "6k-8k"),
        (8192, "8k+"),
        (99999, "8k+"),
    ],
)
def test_token_bucket_labels(value, expected):
    assert ev.token_bucket(value) == expected


def test_token_buckets_returns_ordered_categorical():
    cat = ev.token_buckets([None, 100, 5000, 9000])
    # Ordered categorical so groupby + plots keep the bucket ordering stable
    # regardless of which buckets are populated.
    assert list(cat) == ["unknown", "<2k", "4k-6k", "8k+"]
    assert cat.ordered
    assert list(cat.categories) == ["<2k", "2k-4k", "4k-6k", "6k-8k", "8k+", "unknown"]


def test_load_pool_token_fields(seeded_study):
    cfg, scenarios = seeded_study
    s1 = next(s for s in scenarios if s.id == "S1")
    pool = ev.load_scenario_pool(cfg, s1)
    assert pool.n_tokens.dtype == np.int32
    assert pool.n_tokens.shape == (4,)
    # Per-chunk tuple length aligns with n_chunks for every record.
    assert len(pool.n_tokens_per_chunk) == 4
    for i in range(4):
        assert len(pool.n_tokens_per_chunk[i]) == int(pool.n_chunks[i])
    # de-2 lands in the 8k+ bucket per fixture choice.
    de2_idx = pool.index_of("de-2")
    assert de2_idx is not None
    assert int(pool.n_tokens[de2_idx]) == 9100


def test_load_pool_falls_back_when_token_fields_missing(tmp_path, monkeypatch):
    """Older shards predate the fields; loader should still produce a usable pool."""
    cfg = _make_study_config(tmp_path)
    scenario = Scenario(
        id="S0",
        label="truncate-8190",
        chunker_name=None,
        chunk_tokens=None,
        aggregator_name=None,
    )
    bases = np.eye(2, dtype=np.float32)
    legacy_records = [
        {
            "ci_id": "x",
            "lg": "fr",
            "n_chunks": 1,
            "embedding": _unit(bases[0].tolist()),
            "size": 2,
        },
    ]
    _write_jsonl_bz2(
        cfg.local_path(scenario_filename(scenario.id)), legacy_records
    )
    monkeypatch.setattr(
        ev, "ensure_local", lambda study_cfg, filename, *, force=False: study_cfg.local_path(filename)
    )
    pool = ev.load_scenario_pool(cfg, scenario)
    # Missing fields → 0 total + a single-bucket tuple of (0,), so consumers
    # can iterate without None-guards.
    assert int(pool.n_tokens[0]) == 0
    assert pool.n_tokens_per_chunk[0] == (0,)


def test_score_queries_includes_n_tokens_and_bucket(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    assert "n_tokens" in df.columns
    assert "token_bucket" in df.columns
    # Each query's n_tokens equals the seed value for its gold doc, regardless
    # of scenario (n_tokens is a doc property, not a chunker property).
    fr1 = df[df.ci_id == "fr-1"].drop_duplicates("scenario_id")
    assert (fr1["n_tokens"] == 3500).all()
    assert (fr1["token_bucket"].astype(str) == "2k-4k").all()
    de2 = df[df.ci_id == "de-2"].drop_duplicates("scenario_id")
    assert (de2["n_tokens"] == 9100).all()
    assert (de2["token_bucket"].astype(str) == "8k+").all()


def test_truncation_loss_table_quantifies_s0_loss(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    table = ev.truncation_loss_table(df, baseline_id="S0", limit=8190)
    # Long-form: one row per (lg, token_bucket) with at least one S0 query.
    # de-2 alone is in 8k+; S0 has one query per gold doc, so the de + 8k+
    # bucket is non-empty and 100% truncated.
    de_8k = table[(table["lg"] == "de") & (table["token_bucket"].astype(str) == "8k+")]
    assert len(de_8k) == 1
    assert float(de_8k.iloc[0]["pct_truncated"]) == pytest.approx(1.0)
    assert float(de_8k.iloc[0]["mean_truncated_tokens"]) == pytest.approx(9100 - 8190)
    # fr-1 in the 2k-4k bucket is well under 8190 → 0% truncated.
    fr_2k = table[(table["lg"] == "fr") & (table["token_bucket"].astype(str) == "2k-4k")]
    assert len(fr_2k) == 1
    assert float(fr_2k.iloc[0]["pct_truncated"]) == pytest.approx(0.0)


def test_truncation_loss_table_empty_when_baseline_absent(seeded_study):
    cfg, scenarios = seeded_study
    inputs = ev.load_eval_inputs(cfg, scenarios=scenarios)
    df = ev.score_queries(inputs)
    # Strip S0 rows so the helper has no baseline to filter to.
    df_no_baseline = df[df["scenario_id"] != "S0"]
    table = ev.truncation_loss_table(df_no_baseline, baseline_id="S0")
    assert table.empty
