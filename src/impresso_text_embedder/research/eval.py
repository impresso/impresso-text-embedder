"""Eval harness for the chunking-strategy sweep.

Step 8 of ``research/chunking-eval``: turn the per-scenario doc
embeddings (from :mod:`research.embed_sweep`) and the per-query
embeddings (from :mod:`research.query_embed`) into a tidy
DataFrame of retrieval scores, plus the loader / sanity-check
plumbing the per-study notebook consumes.

The notebook (``notebooks/<study>-eval.ipynb``) is the analysis
surface; this module owns everything that benefits from being
tested in isolation:

- :func:`load_eval_inputs` — pull the queries-embedded shard plus
  every scenario's doc-embedding shard down from S3 into a frozen
  :class:`EvalInputs`. Files cache under
  ``study_cfg.local_path(...)`` so re-running a notebook cell
  doesn't re-download.
- :func:`run_sanity` — assertions the notebook prints before
  scoring (schema match, dim, lg coverage, ci_id alignment).
- :func:`score_queries` — per-(query, scenario) retrieval scores
  (rank of the gold doc, ``recall@k`` flags, MRR contribution) as
  a tidy :class:`pandas.DataFrame`.
- :func:`bootstrap_ci` — percentile bootstrap for confidence
  intervals on aggregated metrics.

Per-language retrieval pool: every query ranks against the
same-language slice of the scenario's doc embeddings (fr queries
vs fr docs only). Cross-lingual ranking is the gated O4 ablation
and is intentionally out of scope here.

Chunk-level metrics (IoU / Precision_Ω over chunk-vs-reference
spans) are deferred — they require a chunker-span-recovery helper
that doesn't exist on this branch yet. The doc-level metrics in
this module fully answer the headline research question (does
forced sub-context chunking change doc-level embedding quality?).

Design narrative + rejected alternatives in
``.progress/eval-harness/notes.md``.
"""

from __future__ import annotations

import bz2
import dataclasses
import logging
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from impresso_text_embedder import io as s3io
from impresso_text_embedder.research.scenario_builder import ScenarioRegistry
from impresso_text_embedder.research.scenarios import Scenario
from impresso_text_embedder.research.study_config import (
    QUERIES_EMBEDDED_FILENAME,
    StudyConfig,
    scenario_filename,
)

log = logging.getLogger(__name__)

# Default ranks for Recall@k columns. Keeping it tight (1/5/10)
# matches the IR-eval convention and avoids parquet column bloat.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10)

# Matryoshka prefix dimensions for ``gte-multilingual-base``. The model is
# trained with Matryoshka heads at these targets, so a prefix of length
# ``d ∈ DEFAULT_TRUNCATION_DIMS`` is itself a meaningful embedding once
# re-L2-normalised. ``768`` is the full hidden size; smaller values trade
# storage / cosine cost against retrieval quality. Notebook-level constant —
# the helper itself is dim-agnostic.
DEFAULT_TRUNCATION_DIMS: tuple[int, ...] = (64, 128, 256, 512, 768)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QueryRow:
    """One per-query record from the queries-embedded shard."""

    query_id: str
    ci_id: str
    lg: str
    query_type: str
    position_bucket: str
    position_chars: tuple[int, int]
    references: tuple[dict[str, Any], ...]
    query_text: str
    embedding: np.ndarray  # shape (dim,), float32, unit-norm


@dataclasses.dataclass(frozen=True)
class ScenarioPool:
    """Per-scenario doc embedding pool keyed by ``ci_id``.

    All arrays share the same row order: row ``i`` corresponds to
    ``ci_ids[i]``. ``embeddings`` are unit-norm per the model's
    built-in ``Normalize`` module + the mean-pool aggregator's
    post-aggregation L2 renorm (CLAUDE.md → "Two independent L2
    normalizations").
    """

    scenario: Scenario
    ci_ids: np.ndarray  # shape (N,), dtype object (str)
    embeddings: np.ndarray  # shape (N, dim), float32
    langs: np.ndarray  # shape (N,), dtype object (str)
    n_chunks: np.ndarray  # shape (N,), int
    n_tokens: np.ndarray  # shape (N,), int32 — exact total tokens per doc
    # Per-chunk token counts as a tuple-of-tuples (length N; inner tuple has
    # ``n_chunks[i]`` ints). Tuples instead of a numpy ragged array keep the
    # dataclass ``frozen=True`` and dodge object-dtype sentinel handling.
    n_tokens_per_chunk: tuple[tuple[int, ...], ...]
    len_chars: np.ndarray  # shape (N,), int64 — needed for chars_per_token (O5)

    def lang_mask(self, lg: str) -> np.ndarray:
        return self.langs == lg

    def index_of(self, ci_id: str) -> int | None:
        """Row index for ``ci_id`` or ``None`` if absent."""
        hits = np.where(self.ci_ids == ci_id)[0]
        if hits.size == 0:
            return None
        return int(hits[0])


@dataclasses.dataclass(frozen=True)
class EvalInputs:
    """Bundle the notebook needs to run scoring + plotting."""

    study: StudyConfig
    queries: tuple[QueryRow, ...]
    pools: dict[str, ScenarioPool]  # scenario_id -> pool
    scenarios: tuple[Scenario, ...]


@dataclasses.dataclass
class SanityReport:
    """Pre-scoring assertions; the notebook renders this as a table."""

    n_queries: int
    n_query_languages: int
    embedding_dim: int
    pool_size_per_scenario: dict[str, int]
    pool_lg_counts_per_scenario: dict[str, dict[str, int]]
    queries_lg_counts: dict[str, int]
    queries_position_counts: dict[str, int]
    queries_type_counts: dict[str, int]
    queries_with_unmatched_ci_id: list[str]
    issues: list[str]

    @property
    def passed(self) -> bool:
        return not self.issues


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def ensure_local(
    study_cfg: StudyConfig, filename: str, *, force: bool = False
) -> Path:
    """Download ``filename`` from S3, decompress if bz2, cache, return.

    Notebooks run cells repeatedly; the production
    :func:`research._io.staged_input` deletes its tempfile on
    context exit, which is exactly the wrong behaviour. Use this
    instead — the artefact lives under ``study_cfg.local_path(...)``
    and is reused on every call until ``force=True``. For ``*.bz2``
    inputs the bz2 is decompressed to a sibling ``.jsonl`` and the
    compressed copy is removed, so the on-disk cache is grep-able and
    openable without a decompressor.
    """
    bz2_path = study_cfg.local_path(filename)
    decompress = filename.endswith(".bz2")
    local = bz2_path.with_suffix("") if decompress else bz2_path
    if local.exists() and not force:
        log.debug("cache hit: %s", local)
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    s3_key = study_cfg.s3_key(filename)
    log.info("downloading s3://%s/%s -> %s", study_cfg.s3.bucket, s3_key, bz2_path)
    s3io.download_to_local(study_cfg.s3.bucket, s3_key, bz2_path)
    if decompress:
        log.info("decompressing %s -> %s", bz2_path, local)
        with bz2.open(bz2_path, "rb") as src, open(local, "wb") as dst:
            shutil.copyfileobj(src, dst)
        bz2_path.unlink()
    return local


def _iter_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(orjson.loads(line))
    return out


def _to_query_row(rec: dict) -> QueryRow:
    pos = rec.get("position_chars") or (0, 0)
    embedding = np.asarray(rec["embedding"], dtype=np.float32)
    return QueryRow(
        query_id=str(rec["query_id"]),
        ci_id=str(rec["ci_id"]),
        lg=str(rec.get("lg", "")),
        query_type=str(rec.get("query_type", "")),
        position_bucket=str(rec.get("position_bucket", "")),
        position_chars=(int(pos[0]), int(pos[1])),
        references=tuple(rec.get("references") or ()),
        query_text=str(rec.get("query_text", "")),
        embedding=embedding,
    )


def load_queries(
    study_cfg: StudyConfig, *, force: bool = False
) -> tuple[QueryRow, ...]:
    """Pull and parse ``queries-embedded.jsonl.bz2`` for a study."""
    path = ensure_local(study_cfg, QUERIES_EMBEDDED_FILENAME, force=force)
    return tuple(_to_query_row(r) for r in _iter_jsonl(path))


def load_scenario_pool(
    study_cfg: StudyConfig, scenario: Scenario, *, force: bool = False
) -> ScenarioPool:
    """Pull and parse one scenario's doc-embedding shard."""
    filename = scenario_filename(scenario.id)
    path = ensure_local(study_cfg, filename, force=force)
    records = _iter_jsonl(path)
    if not records:
        raise ValueError(
            f"scenario {scenario.id}: empty embedding shard at {path}"
        )
    n = len(records)
    dim = len(records[0]["embedding"])
    embeddings = np.empty((n, dim), dtype=np.float32)
    ci_ids = np.empty(n, dtype=object)
    langs = np.empty(n, dtype=object)
    n_chunks = np.empty(n, dtype=np.int32)
    n_tokens = np.empty(n, dtype=np.int32)
    len_chars = np.empty(n, dtype=np.int64)
    per_chunk_acc: list[tuple[int, ...]] = []
    for i, r in enumerate(records):
        embeddings[i] = np.asarray(r["embedding"], dtype=np.float32)
        ci_ids[i] = str(r.get("ci_id"))
        langs[i] = str(r.get("lg", ""))
        n_chunks[i] = int(r.get("n_chunks") or 1)
        n_tokens[i] = int(r.get("n_tokens") or 0)
        len_chars[i] = int(r.get("len_chars") or 0)
        per_chunk = r.get("n_tokens_per_chunk")
        # Older shards predate the field; fall back to a single-bucket tuple
        # so consumers can iterate uniformly without None-guards.
        per_chunk_acc.append(
            tuple(int(x) for x in per_chunk)
            if per_chunk
            else (int(n_tokens[i]),)
        )
    if n > 0 and not bool(n_tokens.any()):
        # Every record fell through the missing-field fallback. Either the
        # shard genuinely predates ``n_tokens`` or — far more likely — the
        # local mirror is stale and ``ensure_local`` returned a cached file
        # newer-on-S3-than-on-disk. Loud, not fatal: legacy shards are still
        # readable, but stratification + truncation_loss + chars_per_token
        # calibration will all produce ``"unknown"`` rows.
        log.warning(
            "scenario %s: n_tokens absent from every record at %s — local cache may "
            "predate the field; reload with load_*(force=True) or delete the "
            "local mirror to refresh from S3.",
            scenario.id,
            path,
        )
    return ScenarioPool(
        scenario=scenario,
        ci_ids=ci_ids,
        embeddings=embeddings,
        langs=langs,
        n_chunks=n_chunks,
        n_tokens=n_tokens,
        n_tokens_per_chunk=tuple(per_chunk_acc),
        len_chars=len_chars,
    )


def load_eval_inputs(
    study_cfg: StudyConfig,
    *,
    scenarios: Sequence[Scenario] | None = None,
    force: bool = False,
) -> EvalInputs:
    """Load queries + every scenario pool for a study.

    If ``scenarios`` is ``None``, the full registry derived from the
    study config is loaded — that is the analysis path. Pass an
    explicit subset for smoke runs (e.g. ``[S0, S5]``).
    """
    if scenarios is None:
        scenarios = ScenarioRegistry.from_study(study_cfg).all_scenarios()
    queries = load_queries(study_cfg, force=force)
    pools: dict[str, ScenarioPool] = {}
    for scen in scenarios:
        pools[scen.id] = load_scenario_pool(study_cfg, scen, force=force)
    return EvalInputs(
        study=study_cfg,
        queries=queries,
        pools=pools,
        scenarios=tuple(scenarios),
    )


def _l2_renormalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 renorm; zero rows pass through unchanged.

    ``arr`` may be 1-D (single vector) or 2-D (row-major matrix). Output
    dtype is preserved. Zero-norm rows survive without a NaN — matches
    :func:`cosine_scores`'s defensive style so a degenerate truncation
    prefix doesn't poison the rank pipeline.
    """
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return arr / safe


def truncate_inputs(inputs: EvalInputs, dim: int) -> EvalInputs:
    """Return a copy of ``inputs`` with every embedding sliced to its first
    ``dim`` components and re-L2-normalised.

    Used for Matryoshka-style dim-truncation analysis:
    ``Alibaba-NLP/gte-multilingual-base`` is trained with Matryoshka heads
    at :data:`DEFAULT_TRUNCATION_DIMS`, so a prefix of the 768-d output is
    itself a meaningful unit-norm embedding once renormalised.

    Behaviour:

    - ``dim == full embedding dim`` returns a copy with arrays renormed
      anyway. The renorm is a no-op up to fp32 round-off, and the copy
      lets callers loop over a dim sweep without a special case.
    - ``dim > full dim`` raises :class:`ValueError`. Zero-padding would
      inject a degenerate subspace and silently inflate cosine
      distances, so we refuse rather than guess.
    - ``dim <= 0`` raises :class:`ValueError`.
    - Rows whose first ``dim`` components are all zero (rare; only
      possible for degenerate fixtures) are left at zero — the existing
      :func:`cosine_scores` is defensive against zero vectors and
      :func:`rank_of_gold` will then place them at the bottom of the
      tied-pessimistic rank.

    The returned :class:`EvalInputs` reuses ``study`` and ``scenarios``;
    only ``queries`` and ``pools`` are rebuilt with truncated arrays.
    """
    if dim <= 0:
        raise ValueError(f"truncate_inputs: dim must be > 0, got {dim}")

    embedding_dims = {q.embedding.shape[0] for q in inputs.queries}
    embedding_dims.update(p.embeddings.shape[1] for p in inputs.pools.values())
    if not embedding_dims:
        return inputs
    full_dim = max(embedding_dims)
    if dim > full_dim:
        raise ValueError(
            f"truncate_inputs: dim={dim} exceeds full embedding dim {full_dim}; "
            "zero-padding is refused (would inflate cosine distances)"
        )

    new_queries = tuple(
        dataclasses.replace(
            q,
            embedding=_l2_renormalize(q.embedding[:dim].astype(np.float32, copy=True)),
        )
        for q in inputs.queries
    )
    new_pools: dict[str, ScenarioPool] = {}
    for sid, pool in inputs.pools.items():
        new_pools[sid] = dataclasses.replace(
            pool,
            embeddings=_l2_renormalize(
                pool.embeddings[:, :dim].astype(np.float32, copy=True)
            ),
        )
    return dataclasses.replace(
        inputs, queries=new_queries, pools=new_pools
    )


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def run_sanity(inputs: EvalInputs) -> SanityReport:
    """Pre-scoring assertions. Issues are collected, never raised — the
    notebook prints them and decides whether to halt.
    """
    issues: list[str] = []
    if not inputs.queries:
        issues.append("queries-embedded shard is empty")
    if not inputs.pools:
        issues.append("no scenario pools loaded")

    dims = {q.embedding.shape[0] for q in inputs.queries}
    for pool in inputs.pools.values():
        dims.add(pool.embeddings.shape[1])
    if len(dims) > 1:
        issues.append(f"inconsistent embedding dim across artefacts: {sorted(dims)}")
    embedding_dim = next(iter(dims), 0)

    pool_size = {sid: int(p.embeddings.shape[0]) for sid, p in inputs.pools.items()}
    if len(set(pool_size.values())) > 1:
        issues.append(
            f"scenarios cover different ci_id sets — pool sizes {pool_size}"
        )

    lg_counts: dict[str, dict[str, int]] = {}
    for sid, pool in inputs.pools.items():
        unique, counts = np.unique(pool.langs, return_counts=True)
        lg_counts[sid] = dict(zip(unique.tolist(), counts.tolist(), strict=True))

    queries_lg: dict[str, int] = {}
    queries_bucket: dict[str, int] = {}
    queries_type: dict[str, int] = {}
    for q in inputs.queries:
        queries_lg[q.lg] = queries_lg.get(q.lg, 0) + 1
        queries_bucket[q.position_bucket] = queries_bucket.get(q.position_bucket, 0) + 1
        queries_type[q.query_type] = queries_type.get(q.query_type, 0) + 1

    # ci_id coverage: every query's gold ci_id must appear in every
    # scenario's pool (in the same language). Spot-check the first
    # scenario; mismatches across scenarios surface as pool-size
    # divergence above.
    unmatched: list[str] = []
    if inputs.pools:
        first = next(iter(inputs.pools.values()))
        ci_set = set(first.ci_ids.tolist())
        for q in inputs.queries:
            if q.ci_id not in ci_set:
                unmatched.append(q.query_id)
    if unmatched:
        issues.append(
            f"{len(unmatched)} queries have ci_id absent from the doc pool "
            f"(first 3: {unmatched[:3]})"
        )

    # NaN / non-finite embeddings.
    for q in inputs.queries:
        if not np.isfinite(q.embedding).all():
            issues.append(f"query {q.query_id} has non-finite embedding")
            break
    for sid, pool in inputs.pools.items():
        if not np.isfinite(pool.embeddings).all():
            issues.append(f"scenario {sid} has non-finite embeddings")
            break

    # Unit-norm spot-check (tolerate 1e-3 because of fp32 round-trip
    # plus bf16 encode + 5-dp rounding on disk).
    norms = np.linalg.norm(
        np.stack([q.embedding for q in inputs.queries[: min(8, len(inputs.queries))]]),
        axis=1,
    )
    if inputs.queries and not np.allclose(norms, 1.0, atol=1e-2):
        issues.append(
            f"query embeddings not unit-norm — sample norms {norms.tolist()}"
        )
    for sid, pool in inputs.pools.items():
        sample = pool.embeddings[: min(8, pool.embeddings.shape[0])]
        sample_norms = np.linalg.norm(sample, axis=1)
        if not np.allclose(sample_norms, 1.0, atol=1e-2):
            issues.append(
                f"scenario {sid} embeddings not unit-norm — "
                f"sample norms {sample_norms.tolist()}"
            )
            break

    return SanityReport(
        n_queries=len(inputs.queries),
        n_query_languages=len({q.lg for q in inputs.queries}),
        embedding_dim=embedding_dim,
        pool_size_per_scenario=pool_size,
        pool_lg_counts_per_scenario=lg_counts,
        queries_lg_counts=queries_lg,
        queries_position_counts=queries_bucket,
        queries_type_counts=queries_type,
        queries_with_unmatched_ci_id=unmatched,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Metrics — pure functions on numpy arrays
# ---------------------------------------------------------------------------


def cosine_scores(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Return the cosine similarity between ``query`` (1D) and each row of
    ``pool`` (2D). Both are expected unit-norm; we still divide by norms
    defensively so callers can pass non-unit input without surprise.
    """
    query = np.asarray(query, dtype=np.float32)
    pool = np.asarray(pool, dtype=np.float32)
    q_norm = np.linalg.norm(query)
    p_norms = np.linalg.norm(pool, axis=1)
    if q_norm == 0 or np.any(p_norms == 0):
        # Defensive: 0 vectors give 0 similarity, no NaN.
        denom = np.where(p_norms == 0, 1.0, p_norms) * (q_norm or 1.0)
        return (pool @ query) / denom
    return (pool @ query) / (q_norm * p_norms)


def rank_of_gold(scores: np.ndarray, gold_idx: int) -> int:
    """Return the 1-based rank of ``gold_idx`` under ``scores``.

    Ties broken pessimistically: an item tied with k other items that
    score higher *or equal* gets rank ``k+1``. Equivalent to the
    competition-rank ("1224") convention. Pessimistic ties prevent a
    chunker that produces many identical near-zero embeddings from
    inflating its Recall@1.
    """
    gold_score = scores[gold_idx]
    higher_or_equal = int(np.sum(scores >= gold_score)) - 1  # exclude self
    return higher_or_equal + 1

def margin_of_gold(scores: np.ndarray, gold_idx: int) -> float:
    """Return the margin between the gold score and the max non-gold score.

    Positive means the gold is ahead of all competitors; negative means
    it's behind at least one. This is a more fine-grained metric than rank
    that still captures the "win/lose" aspect of retrieval.
    """
    if scores.size <= 1:
        return float("nan")  # no competitors, margin undefined
    gold_score = float(scores[gold_idx])
    mask = np.ones(scores.shape[0], dtype=bool)
    mask[gold_idx] = False
    max_competitor = float(np.max(scores[mask]))
    return gold_score - max_competitor

def softmax_nll(scores: np.ndarray, gold_idx: int, temperature: float = 0.05) -> float:
    """Return the negative log-likelihood of the gold under a softmax over scores."""
    if scores.size == 0:
        return float("nan")  # no competitors, NLL undefined
    # Apply temperature to the scores
    logits = scores.astype(np.float64) / temperature
    m = float(logits.max())
    log_z = m + math.log(float(np.exp(logits - m).sum()))
    return log_z - float(logits[gold_idx]) 


def recall_at_k(ranks: Sequence[int] | np.ndarray, k: int) -> float:
    """Mean fraction of queries with ``rank <= k``."""
    arr = np.asarray(ranks, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr <= k))


def reciprocal_ranks(ranks: Sequence[int] | np.ndarray) -> np.ndarray:
    """``1/rank`` — agnostic to NaN (preserved as NaN)."""
    arr = np.asarray(ranks, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = 1.0 / arr
    rr[np.isnan(arr)] = np.nan
    return rr


def mean_reciprocal_rank(ranks: Sequence[int] | np.ndarray) -> float:
    rr = reciprocal_ranks(ranks)
    if rr.size == 0:
        return float("nan")
    return float(np.nanmean(rr))


# Default bucket edges tuned for Study A's (4096, 8192) corpus. Studies
# spanning a different range (e.g. Study B at 16k+) pass their own.
DEFAULT_TOKEN_BUCKET_EDGES: tuple[int, ...] = (2048, 4096, 6144, 8192)


def _is_missing(value: Any) -> bool:
    """True for ``NaN`` (float) or ``pd.NA``; False for plain ints / floats."""
    try:
        # ``pd.NA == pd.NA`` is ``pd.NA`` (truthy-bool raises), so we go via
        # ``math.isnan`` after a finite-cast guard.
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _token_bucket_labels(edges: Sequence[int]) -> list[str]:
    """Stable bucket labels matching ``edges`` so plot ordering is fixed."""
    labels = [f"<{edges[0] // 1024}k"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"{lo // 1024}k-{hi // 1024}k")
    labels.append(f"{edges[-1] // 1024}k+")
    return labels


def token_bucket(
    n_tokens: int | float | None,
    *,
    edges: Sequence[int] = DEFAULT_TOKEN_BUCKET_EDGES,
) -> str:
    """Return a stable string label for a single ``n_tokens`` value.

    Returns ``"unknown"`` for ``None`` and 0 (the loader's missing-field
    fallback for older shards) so the bucket axis cleanly separates "no
    data" from "<2k". Edges are right-exclusive: a value at the edge
    lands in the upper bucket (e.g. 2048 → ``"2k-4k"``).
    """
    # NaN survives boxing through pandas (``None`` → ``NaN`` in object/numeric
    # columns) and ``pd.NA`` shows up from ``Int64`` columns; treat them all
    # as "unknown" so the lambda used by :func:`token_buckets` is total.
    if n_tokens is None or _is_missing(n_tokens) or n_tokens == 0:
        return "unknown"
    n = int(n_tokens)
    labels = _token_bucket_labels(edges)
    for i, edge in enumerate(edges):
        if n < edge:
            return labels[i]
    return labels[-1]


def token_buckets(
    values: Any,
    *,
    edges: Sequence[int] = DEFAULT_TOKEN_BUCKET_EDGES,
) -> Any:
    """Vectorised :func:`token_bucket` returning an ordered ``pd.Categorical``.

    Pandas import is lazy so the rest of the module stays usable without
    pandas. Ordered categorical preserves bucket order across groupbys
    and seaborn facets — otherwise plots reorder lexicographically and
    "<2k" lands between "2k-4k" and "8k+".
    """
    import pandas as pd

    labels = _token_bucket_labels(edges) + ["unknown"]
    arr = pd.Series(values).map(lambda v: token_bucket(v, edges=edges))
    return pd.Categorical(arr, categories=labels, ordered=True)


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    statistic: Any = np.mean,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``values``.

    Returns ``(point_estimate, low, high)`` where ``low``/``high`` are
    the symmetric percentile bounds at confidence level ``ci``. Used
    by the notebook to decorate bar plots with error bars and by the
    verdict table to flag scenarios whose CI excludes the S0 baseline.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    samples = rng.choice(arr, size=(n_iter, arr.size), replace=True)
    boots = np.apply_along_axis(statistic, 1, samples)
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boots, alpha))
    high = float(np.quantile(boots, 1.0 - alpha))
    return point, low, high


def paired_bootstrap_delta(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    statistic: Any = np.mean,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for ``statistic(a) - statistic(b)``.

    ``a`` and ``b`` must align row-for-row (same query order). Returns
    ``(delta_point, low, high)``. The notebook uses this to decide
    whether a scenario's lift over S0 is significant: ``low > 0`` means
    it beats the baseline at the chosen CI.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired bootstrap requires aligned shapes, got {a_arr.shape} vs {b_arr.shape}"
        )
    mask = ~(np.isnan(a_arr) | np.isnan(b_arr))
    a_arr, b_arr = a_arr[mask], b_arr[mask]
    if a_arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = float(statistic(a_arr) - statistic(b_arr))
    n = a_arr.size
    idx = rng.integers(0, n, size=(n_iter, n))
    boots = np.array(
        [statistic(a_arr[row]) - statistic(b_arr[row]) for row in idx]
    )
    alpha = (1.0 - ci) / 2.0
    return point, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------


def score_queries(
    inputs: EvalInputs,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
):
    """Return a tidy DataFrame of per-(query, scenario) retrieval scores.

    Pandas import is lazy — the rest of this module stays importable
    in environments without pandas (e.g. a CPU-only test runner).

    Columns:

    - ``query_id``, ``ci_id``, ``lg``, ``query_type``,
      ``position_bucket`` — query identity / stratification keys.
    - ``scenario_id``, ``scenario_label``, ``chunker``,
      ``chunk_tokens``, ``aggregator`` — scenario stratification keys.
      ``aggregator`` is ``"(none)"`` for the truncate baseline; for
      multi-aggregator studies (e.g. ``C-aggregator``) it lets a
      ``chunk_tokens × aggregator`` heatmap fall out of one groupby.
    - ``pool_size`` — same-language pool the rank is computed against.
    - ``rank`` — 1-based rank of gold ci_id (NaN if absent from pool).
    - ``reciprocal_rank`` — ``1/rank`` (NaN if absent).
    - ``recall_at_<k>`` — bool per k in ``k_values``.
    - ``n_chunks`` — chunk count for the gold doc under this scenario;
      diagnostic, not a metric.
    - ``n_tokens`` — exact total tokens of the gold doc under this
      scenario; stratification key (also drives ``token_bucket``).
    - ``token_bucket`` — ordered categorical label derived from
      ``n_tokens`` (edges in :data:`DEFAULT_TOKEN_BUCKET_EDGES`); pass
      directly to ``aggregate_recall_with_ci(by=...)`` for length-
      stratified Recall@k.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    # Cache per-(scenario, lg) sub-pool slices once.
    sub_pool_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    # Per-(scenario, ci_id) gold-doc metadata (n_chunks, n_tokens, len_chars).
    # ``None`` entries mean the ci_id is absent from this scenario's pool.
    doc_meta_cache: dict[tuple[str, str], tuple[int, int, int] | None] = {}

    for scen in inputs.scenarios:
        pool = inputs.pools.get(scen.id)
        if pool is None:
            continue
        for q in inputs.queries:
            key = (scen.id, q.lg)
            if key not in sub_pool_cache:
                mask = pool.lang_mask(q.lg)
                sub_pool_cache[key] = (pool.ci_ids[mask], pool.embeddings[mask])
            sub_ci_ids, sub_embeds = sub_pool_cache[key]
            pool_size = int(sub_embeds.shape[0])

            cache_key = (scen.id, q.ci_id)
            if cache_key not in doc_meta_cache:
                idx_full = pool.index_of(q.ci_id)
                if idx_full is None:
                    doc_meta_cache[cache_key] = None
                else:
                    doc_meta_cache[cache_key] = (
                        int(pool.n_chunks[idx_full]),
                        int(pool.n_tokens[idx_full]),
                        int(pool.len_chars[idx_full]),
                    )
            meta = doc_meta_cache[cache_key]

            row: dict[str, Any] = {
                "query_id": q.query_id,
                "ci_id": q.ci_id,
                "lg": q.lg,
                "query_type": q.query_type,
                "position_bucket": q.position_bucket,
                "scenario_id": scen.id,
                "scenario_label": scen.label,
                "chunker": scen.chunker_name or "truncate",
                "chunk_tokens": scen.chunk_tokens,
                "aggregator": scen.aggregator_name or "(none)",
                "pool_size": pool_size,
                "n_chunks": meta[0] if meta is not None else None,
                "n_tokens": meta[1] if meta is not None else None,
                "len_chars": meta[2] if meta is not None else None,
            }
            gold_in_sub = np.where(sub_ci_ids == q.ci_id)[0]
            if pool_size == 0 or gold_in_sub.size == 0:
                row["rank"] = math.nan
                row["reciprocal_rank"] = math.nan
                for k in k_values:
                    row[f"recall_at_{k}"] = False
            else:
                gold_idx = int(gold_in_sub[0])
                scores = cosine_scores(q.embedding, sub_embeds)
                rank = rank_of_gold(scores, gold_idx)
                row["rank"] = rank
                row["reciprocal_rank"] = 1.0 / rank
                row["margin"] = margin_of_gold(scores, gold_idx)
                row["nll"]    = softmax_nll(scores, gold_idx)
                for k in k_values:
                    row[f"recall_at_{k}"] = bool(rank <= k)
            rows.append(row)

    df = pd.DataFrame.from_records(rows)
    if not df.empty:
        df["chunk_tokens"] = df["chunk_tokens"].astype("Int64")
        df["rank"] = df["rank"].astype("Float64")
        df["reciprocal_rank"] = df["reciprocal_rank"].astype("Float64")
        df["margin"] = df["margin"].astype("Float64")
        df["nll"]    = df["nll"].astype("Float64")
        df["n_chunks"] = df["n_chunks"].astype("Int64")
        df["n_tokens"] = df["n_tokens"].astype("Int64")
        df["len_chars"] = df["len_chars"].astype("Int64")
        df["token_bucket"] = token_buckets(df["n_tokens"])
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers (notebook-side)
# ---------------------------------------------------------------------------


def aggregate_recall_with_ci(
    scores_df: Any,
    *,
    metric: str = "recall_at_5",
    by: Sequence[str] = (
        "lg", "scenario_id", "scenario_label", "chunker", "chunk_tokens", "aggregator",
    ),
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
):
    """Group ``scores_df`` and report mean + bootstrap CI for ``metric``.

    Returns a DataFrame with columns ``[*by, point, low, high, n]`` —
    the shape :func:`seaborn.barplot` consumes via ``y='point'`` plus
    manual ``yerr=(point-low, high-point)`` (or ``errwidth=`` if
    asymmetric error bars are over-engineering).
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for keys, group in scores_df.groupby(list(by), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[metric].astype(float).to_numpy()
        point, low, high = bootstrap_ci(
            values, n_iter=n_iter, ci=ci, seed=seed
        )
        rec = dict(zip(by, keys, strict=True))
        rec.update({"point": point, "low": low, "high": high, "n": int(values.size)})
        rows.append(rec)
    return pd.DataFrame.from_records(rows)


def baseline_delta_table(
    scores_df: Any,
    *,
    metric: str = "recall_at_5",
    baseline_id: str = "S0",
    by: Sequence[str] = (
        "lg", "scenario_id", "scenario_label", "chunker", "chunk_tokens", "aggregator",
    ),
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
):
    """Per-scenario Δ vs ``baseline_id`` with paired-bootstrap CI.

    The verdict table — sort by ``delta`` descending and look at the
    ``low`` column to spot scenarios whose CI excludes 0.
    """
    import pandas as pd

    cols = list(by) + ["query_id"]
    pivot = scores_df.pivot_table(
        index="query_id",
        columns="scenario_id",
        values=metric,
        aggfunc="first",
    )
    if baseline_id not in pivot.columns:
        raise ValueError(f"baseline scenario {baseline_id!r} absent from scores")

    base = pivot[baseline_id].astype(float)
    rows: list[dict[str, Any]] = []
    scen_meta = (
        scores_df[list(by)].drop_duplicates(subset=["scenario_id"]).set_index("scenario_id")
    )
    for sid in pivot.columns:
        if sid == baseline_id:
            continue
        a = pivot[sid].astype(float).to_numpy()
        b = base.to_numpy()
        delta, low, high = paired_bootstrap_delta(
            a, b, n_iter=n_iter, ci=ci, seed=seed
        )
        meta = scen_meta.loc[sid].to_dict()
        meta["scenario_id"] = sid
        meta["delta"] = delta
        meta["low"] = low
        meta["high"] = high
        meta["beats_baseline"] = bool(low > 0)
        meta["loses_to_baseline"] = bool(high < 0)
        rows.append(meta)
    out = pd.DataFrame.from_records(rows)
    if not out.empty:
        out = out.sort_values("delta", ascending=False).reset_index(drop=True)
    return out


def truncation_loss_table(
    scores_df: Any,
    *,
    baseline_id: str = "S0",
    limit: int = 8190,
    metric: str = "recall_at_5",
):
    """Diagnostic: how much content does the truncate baseline silently drop?

    Filters to ``baseline_id`` rows (one per (query, S0) pair), bins by
    ``(lg, token_bucket)``, and reports per bucket: mean ``metric``, MRR,
    fraction of rows whose gold doc exceeded ``limit`` tokens, and the mean
    truncated-token count over that fraction. Lets the notebook quantify
    "what does S0 lose, and where?" without re-running the encode path.
    """
    import pandas as pd

    base = scores_df[scores_df["scenario_id"] == baseline_id].copy()
    if base.empty:
        return pd.DataFrame()
    base["truncated_tokens"] = (base["n_tokens"].astype("Int64") - limit).clip(lower=0)
    base["was_truncated"] = base["truncated_tokens"] > 0

    rows: list[dict[str, Any]] = []
    for keys, group in base.groupby(["lg", "token_bucket"], dropna=False, observed=False):
        lg, bucket = keys if isinstance(keys, tuple) else (keys, None)
        m = group[metric].astype(float).to_numpy()
        ranks = group["rank"].astype(float).to_numpy()
        rows.append(
            {
                "lg": lg,
                "token_bucket": bucket,
                "n": int(group.shape[0]),
                metric: float(np.nanmean(m)) if m.size else float("nan"),
                "mrr": mean_reciprocal_rank(ranks),
                "pct_truncated": float(group["was_truncated"].mean()),
                "mean_truncated_tokens": float(
                    group.loc[group["was_truncated"], "truncated_tokens"].astype(float).mean()
                ) if group["was_truncated"].any() else 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)


__all__ = [
    "DEFAULT_K_VALUES",
    "DEFAULT_TOKEN_BUCKET_EDGES",
    "DEFAULT_TRUNCATION_DIMS",
    "EvalInputs",
    "QueryRow",
    "SanityReport",
    "ScenarioPool",
    "aggregate_recall_with_ci",
    "baseline_delta_table",
    "bootstrap_ci",
    "cosine_scores",
    "ensure_local",
    "load_eval_inputs",
    "load_queries",
    "load_scenario_pool",
    "mean_reciprocal_rank",
    "paired_bootstrap_delta",
    "rank_of_gold",
    "recall_at_k",
    "reciprocal_ranks",
    "run_sanity",
    "score_queries",
    "token_bucket",
    "token_buckets",
    "truncate_inputs",
    "truncation_loss_table",
]
