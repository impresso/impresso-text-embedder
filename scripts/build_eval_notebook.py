"""Generate ``notebooks/<study>-eval.ipynb`` from a single source of truth.

The notebook is the deliverable for one study's chunking-eval
report (step 8 of ``research/chunking-eval``). One notebook per
study; the cell list is identical across studies and only the
``STUDY_CONFIG_PATH`` constant in the first code cell differs.
This generator owns that template so a tweak to a plot or a
sanity check lands in every per-study notebook with one rerun.

Run::

    uv run python scripts/build_eval_notebook.py study-A-fit
    uv run python scripts/build_eval_notebook.py study-B-overflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
CONFIGS_DIR = REPO_ROOT / "configs/research"


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def build_notebook(study: str) -> nbformat.NotebookNode:
    cfg_rel = f"configs/research/{study}.yaml"
    cells = [
        md(
            f"""
# Chunking-eval report — `{study}`

**Branch**: `research/chunking-eval` · **Step**: 8 (eval-harness-notebook)

This notebook is the deliverable for one study of the chunking-strategy sweep.
It loads the per-scenario doc-embedding shards (from `embed_sweep`) and the
per-query embedding shard (from `query_embed`) for `{study}`, computes
doc-level retrieval scores per `(query, scenario)`, and renders the plots and
tables that answer the headline research question:

> *Does forcing sub-context chunking + mean+L2 pool produce better doc-level
> embeddings than one-shot encoding the whole doc?*

## How to read this notebook

1. **Setup → Sanity** (sections 1–3): load inputs, verify schema/dim/lg
   alignment. Halt and fix upstream if the sanity table flags issues.
2. **Headline verdict** (section 4): one table per language showing the Δ in
   `Recall@5` vs the `S0` truncate baseline, with paired-bootstrap CIs.
3. **Plots** (sections 5–10): Recall@5 by scenario, convergence vs
   `chunk_tokens`, position-bucket robustness, MRR, query-type ablation,
   diagnostic distributions.
4. **Notes** (section 11): per-language conclusion + open follow-ups. **Edit
   this cell when re-running.**

## Scope

- **Embedding level**: `text` only (one vector per doc; mean+L2 when chunked).
  Chunk-level retrieval is out of scope on this branch.
- **Languages**: per-study (see `corpus.languages` in the YAML). Each query
  ranks against the same-language slice of the doc pool only — cross-lingual
  is the gated O4 ablation.
- **Metrics**: Recall@1/5/10, MRR; bootstrap CIs (B=1000) per metric per
  scenario; paired bootstrap for Δ vs the S0 baseline. Chunk-level IoU /
  Precision_Ω deferred (requires chunker span recovery — see notes folder).
"""
        ),
        md("## 1 — Setup"),
        code(
            f"""
# This notebook runs end-to-end given a study YAML config. Change the
# constant below to switch studies; everything else derives from it.
import os
import sys
from pathlib import Path

# When running from the notebook (cwd = notebooks/), reach the src/ layout.
REPO_ROOT = Path(os.environ.get("IMPRESSO_REPO_ROOT", Path.cwd().resolve())).resolve()
if (REPO_ROOT / "notebooks").is_dir() is False and (REPO_ROOT.parent / "notebooks").is_dir():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

STUDY_CONFIG_PATH = REPO_ROOT / "{cfg_rel}"
assert STUDY_CONFIG_PATH.exists(), f"missing: {{STUDY_CONFIG_PATH}}"
print(f"study config: {{STUDY_CONFIG_PATH}}")
"""
        ),
        code(
            """
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from impresso_text_embedder.research import eval as ev
from impresso_text_embedder.research.scenario_builder import ScenarioRegistry
from impresso_text_embedder.research.study_config import load_study_config

# Minimal-noise log surface so notebook output stays readable.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("impresso_text_embedder").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("s3transfer").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Best-practice seaborn theme: white grid for ratio plots, colorblind palette
# (categorical axis = chunker family), tight context. dpi bumped so PNGs in
# the rendered notebook are readable at GitHub's render width without
# blowing up the file size.
sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="colorblind",
    rc={"figure.dpi": 110, "savefig.dpi": 130, "axes.titleweight": "bold"},
)
"""
        ),
        code(
            """
study_cfg = load_study_config(STUDY_CONFIG_PATH)
registry = ScenarioRegistry.from_study(study_cfg)
print(f"study={study_cfg.study.name}  config_sha={study_cfg.config_sha}")
print(f"languages={study_cfg.corpus.languages}  n_per_lg={study_cfg.corpus.n_per_lg}")
print(f"scenarios ({len(registry.all_ids())}):")
print(registry.format_table())
"""
        ),
        md(
            """
## 2 — Load eval inputs

Pulls the queries-embedded shard plus every scenario shard for this study
from `s3://{study_cfg.s3.bucket}/{study_cfg.study_s3_root()}/`. Files cache
under the local mirror (`study_cfg.local_path(...)`); re-running the cell
hits the cache. Pass `force=True` to invalidate.
"""
        ),
        code(
            """
inputs = ev.load_eval_inputs(study_cfg)
print(
    f"loaded {len(inputs.queries)} queries × {len(inputs.pools)} scenarios "
    f"({sum(p.embeddings.shape[0] for p in inputs.pools.values())} doc rows total)"
)
"""
        ),
        md(
            """
## 3 — Sanity checks

Every assertion here is a load-time guard against an upstream pipeline drift
that would silently invalidate the eval (mismatched embedding dim, scenarios
covering different `ci_id` sets, queries whose gold doc is absent, non-finite
or non-unit-norm embeddings). The `Issues` row is empty when everything is
green.
"""
        ),
        code(
            """
report = ev.run_sanity(inputs)

summary_rows = [
    ("queries (total)", report.n_queries),
    ("query languages", report.n_query_languages),
    ("embedding dim", report.embedding_dim),
    ("queries by lg", report.queries_lg_counts),
    ("queries by position bucket", report.queries_position_counts),
    ("queries by query_type", report.queries_type_counts),
    ("pool size per scenario", report.pool_size_per_scenario),
    ("pool lg counts (S0)", report.pool_lg_counts_per_scenario.get("S0", {})),
    ("queries with unmatched ci_id", len(report.queries_with_unmatched_ci_id)),
    ("Issues", report.issues or "(none)"),
]
sanity_df = pd.DataFrame(summary_rows, columns=["check", "value"])
sanity_df.style.set_properties(**{"text-align": "left"}).set_table_styles(
    [{"selector": "th", "props": [("text-align", "left")]}]
)
"""
        ),
        code(
            """
assert report.passed, f"sanity failed: {report.issues}"
"""
        ),
        md(
            """
## 4 — Score every (query, scenario)

`score_queries` joins each query against the same-language slice of every
scenario's doc pool, computes the cosine rank of the gold `ci_id`, and emits
a tidy DataFrame with one row per `(query_id, scenario_id)`. Persisted to
parquet so re-running the plotting cells doesn't re-score.
"""
        ),
        code(
            """
scores = ev.score_queries(inputs, k_values=(1, 5, 10))
out_dir = study_cfg.study_local_root() / "eval"
out_dir.mkdir(parents=True, exist_ok=True)
parquet_path = out_dir / "scores.parquet"
scores.to_parquet(parquet_path, index=False)
print(f"wrote {len(scores):,} score rows -> {parquet_path}")
scores.head()
"""
        ),
        code(
            """
# Order scenarios by registry position so every plot keeps the same x-axis.
SCENARIO_ORDER = registry.all_ids()
SCENARIO_LABELS = {s.id: f"{s.id} · {s.label}" for s in registry.all_scenarios()}
CHUNKER_ORDER = ["truncate", "fixed-window", "token-budget", "semantic"]
LG_ORDER = list(study_cfg.corpus.languages)

# Long display labels render better on rotated x-axes than ids alone.
scores["scenario_display"] = scores["scenario_id"].map(SCENARIO_LABELS)
scenario_display_order = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
"""
        ),
        md(
            """
## 4b — Recall@5 stratified by document token length

The headline answer is almost certainly *length-dependent* — chunking is meant
to help on long docs, and the corpus spans a wide token range. This cell
buckets every score row by `n_tokens` (the exact pre-chunk doc-length, now in
the embed-sweep output) and plots Recall@5 per bucket per scenario. A scenario
that lifts the headline only because it improves the longest-doc band is a
different finding than one that lifts uniformly — read this *before* the
verdict in section 5.
"""
        ),
        code(
            """
g = sns.catplot(
    data=scores,
    x="scenario_display",
    y="recall_at_5",
    hue="token_bucket",
    palette="viridis",
    col="lg",
    col_order=LG_ORDER,
    order=scenario_display_order,
    kind="bar",
    height=4.5,
    aspect=1.7,
    errorbar=("ci", 95),
)
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45, ha="right")
g.set_axis_labels("scenario", "Recall@5")
for ax in g.axes.flat:
    ax.set_ylim(0, 1.02)
g.figure.suptitle(
    f"Token-bucket stratified Recall@5 — study {study_cfg.study.name}",
    y=1.02, fontsize=12,
)
plt.show()
"""
        ),
        md(
            """
## 5 — Headline: Δ vs the truncate baseline (S0)

The verdict table. For each non-baseline scenario we compute the per-query Δ
in `Recall@5` against `S0` (paired bootstrap, B=1000, 95 % CI). Sort by Δ
descending; rows where `low > 0` beat the baseline at the chosen CI; rows
where `high < 0` underperform.

**This is the cell to read first.** Everything below it is supporting evidence.
"""
        ),
        code(
            """
delta_tables = {}
for lg in LG_ORDER:
    sub = scores[scores.lg == lg]
    delta_tables[lg] = ev.baseline_delta_table(
        sub,
        metric="recall_at_5",
        baseline_id="S0",
        n_iter=1000,
        ci=0.95,
    )

for lg, table in delta_tables.items():
    print(f"\\n=== {lg} — Δ Recall@5 vs S0 ===")
    show = table[
        ["scenario_id", "scenario_label", "chunker", "chunk_tokens",
         "delta", "low", "high", "beats_baseline", "loses_to_baseline"]
    ].copy()
    for col in ("delta", "low", "high"):
        show[col] = show[col].astype(float).map(lambda v: f"{v:+.3f}")
    print(show.to_string(index=False))
"""
        ),
        md(
            """
## 6 — Recall@5 by scenario

The headline plot: per-language Recall@5 with bootstrap 95 % CIs, S0 dashed
as the reference line. Bars colored by chunker family so a family-wide
pattern (e.g. token-budget always beats fixed-window at this size) is
visible at a glance.
"""
        ),
        code(
            """
def _palette_for_chunkers():
    base = sns.color_palette("colorblind", n_colors=len(CHUNKER_ORDER))
    return dict(zip(CHUNKER_ORDER, base, strict=True))

chunker_palette = _palette_for_chunkers()

g = sns.catplot(
    data=scores,
    x="scenario_display",
    y="recall_at_5",
    hue="chunker",
    hue_order=CHUNKER_ORDER,
    palette=chunker_palette,
    col="lg",
    col_order=LG_ORDER,
    kind="bar",
    order=scenario_display_order,
    errorbar=("ci", 95),
    height=4.5,
    aspect=1.6,
    dodge=False,  # 1 bar per scenario; hue just colors by family
    legend_out=True,
)
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45, ha="right")
g.set_axis_labels("scenario", "Recall@5")
for lg, ax in g.axes_dict.items():
    s0_value = float(
        scores[(scores.lg == lg) & (scores.scenario_id == "S0")]["recall_at_5"].mean()
    )
    ax.axhline(s0_value, ls="--", color="gray", lw=1, alpha=0.7)
    ax.text(
        0.99, s0_value, f" S0 baseline = {s0_value:.3f}",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        color="gray", fontsize=9,
    )
    ax.set_ylim(0, 1.02)
g.figure.suptitle(
    f"Recall@5 by scenario — study {study_cfg.study.name}",
    y=1.02, fontsize=12,
)
plt.show()
"""
        ),
        md(
            """
## 7 — Convergence: Recall@k vs chunk size

Lines should converge at the largest chunk size — at `chunk_tokens =
model_max_tokens` every chunker collapses toward the truncate baseline.
Divergence at small sizes is the interesting region: a chunker that crashes
at 512 tokens is unsuitable for short-context retrieval.
"""
        ),
        code(
            """
chunked = scores[scores.chunk_tokens.notna()].copy()
chunked["chunk_tokens"] = chunked["chunk_tokens"].astype(int)
g = sns.relplot(
    data=chunked,
    x="chunk_tokens",
    y="recall_at_5",
    hue="chunker",
    hue_order=[c for c in CHUNKER_ORDER if c != "truncate"],
    palette={k: v for k, v in chunker_palette.items() if k != "truncate"},
    col="lg",
    col_order=LG_ORDER,
    kind="line",
    marker="o",
    height=4,
    aspect=1.4,
    errorbar=("ci", 95),
    facet_kws={"sharey": True},
)
for lg, ax in g.axes_dict.items():
    ax.set_xscale("log", base=2)
    s0_value = float(
        scores[(scores.lg == lg) & (scores.scenario_id == "S0")]["recall_at_5"].mean()
    )
    ax.axhline(s0_value, ls="--", color="gray", lw=1, alpha=0.7,
               label=f"S0 baseline ({s0_value:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.02)
g.set_titles("{col_name}")
g.set_axis_labels("chunk_tokens (log₂)", "Recall@5")
g.figure.suptitle(
    f"Convergence — study {study_cfg.study.name}",
    y=1.02, fontsize=12,
)
plt.show()
"""
        ),
        md(
            """
## 8 — Position-bucket robustness

The whole point of step 4's head/mid/tail bucketing: a chunker that always
keeps the head but loses the tail should show a tail-bucket cliff here. For
study-A-fit (docs ≤ context) we expect S0 to be roughly position-flat;
chunkers may differ.
"""
        ),
        code(
            """
position_order = list(study_cfg.query_generation.position_buckets)
g = sns.catplot(
    data=scores,
    x="position_bucket",
    y="recall_at_5",
    hue="chunker",
    hue_order=CHUNKER_ORDER,
    palette=chunker_palette,
    col="lg",
    col_order=LG_ORDER,
    order=position_order,
    kind="bar",
    height=4,
    aspect=1.3,
    errorbar=("ci", 95),
)
g.set_titles("{col_name}")
g.set_axis_labels("position bucket", "Recall@5")
for ax in g.axes.flat:
    ax.set_ylim(0, 1.02)
g.figure.suptitle(
    f"Position-bucket robustness — study {study_cfg.study.name}",
    y=1.02, fontsize=12,
)
plt.show()
"""
        ),
        md(
            """
## 9 — MRR by scenario

MRR is more sensitive than Recall@k to mid-rank shuffling — a scenario that
moves the gold doc from rank 8 to rank 3 looks identical under Recall@10 but
gains MRR. Useful tiebreaker when two scenarios show the same Recall@5.
"""
        ),
        code(
            """
g = sns.catplot(
    data=scores,
    x="scenario_display",
    y="reciprocal_rank",
    hue="chunker",
    hue_order=CHUNKER_ORDER,
    palette=chunker_palette,
    col="lg",
    col_order=LG_ORDER,
    order=scenario_display_order,
    kind="bar",
    estimator=np.nanmean,
    dodge=False,
    height=4.5,
    aspect=1.6,
    errorbar=("ci", 95),
)
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45, ha="right")
g.set_axis_labels("scenario", "MRR")
for lg, ax in g.axes_dict.items():
    s0_value = float(
        scores[(scores.lg == lg) & (scores.scenario_id == "S0")]["reciprocal_rank"]
        .astype(float).mean()
    )
    ax.axhline(s0_value, ls="--", color="gray", lw=1, alpha=0.7)
    ax.set_ylim(0, 1.02)
g.figure.suptitle(
    f"MRR by scenario — study {study_cfg.study.name}",
    y=1.02, fontsize=12,
)
plt.show()
"""
        ),
        md(
            """
## 10 — Query-type ablation

Closes O9 — does `question` retrieve differently than `topical-phrase`? If
one type dominates the signal (e.g. `topical-phrase` is the only metric the
chunker affects), the eval can stratify on it for the verdict.
"""
        ),
        code(
            """
g = sns.catplot(
    data=scores,
    x="scenario_display",
    y="recall_at_5",
    hue="query_type",
    palette="muted",
    col="lg",
    col_order=LG_ORDER,
    order=scenario_display_order,
    kind="bar",
    height=4.5,
    aspect=1.7,
    errorbar=("ci", 95),
)
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45, ha="right")
g.set_axis_labels("scenario", "Recall@5")
for ax in g.axes.flat:
    ax.set_ylim(0, 1.02)
g.figure.suptitle(
    f"Query-type ablation — study {study_cfg.study.name}",
    y=1.02, fontsize=12,
)
plt.show()
"""
        ),
        md(
            """
## 11 — Diagnostics

### 11a — Rank-of-gold distribution

Boxplot per scenario × language. A scenario whose median rank is 1 but whose
upper whisker is at rank 50 is hiding a long tail of hard queries; the
headline Recall@5 doesn't surface that.
"""
        ),
        code(
            """
fig, axes = plt.subplots(
    nrows=len(LG_ORDER), ncols=1,
    figsize=(11, 3.5 * len(LG_ORDER)),
    sharex=True,
)
if len(LG_ORDER) == 1:
    axes = [axes]
for ax, lg in zip(axes, LG_ORDER, strict=True):
    sub = scores[(scores.lg == lg) & scores["rank"].notna()]
    sns.boxplot(
        data=sub,
        x="scenario_display",
        y="rank",
        order=scenario_display_order,
        hue="chunker",
        hue_order=CHUNKER_ORDER,
        palette=chunker_palette,
        ax=ax,
        legend=False,
        dodge=False,
        showfliers=False,
        linewidth=0.8,
    )
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title(f"rank-of-gold distribution — {lg}")
    ax.set_xlabel("")
    ax.set_ylabel("rank (symlog)")
axes[-1].set_xlabel("scenario")
for label in axes[-1].get_xticklabels():
    label.set_rotation(45)
    label.set_horizontalalignment("right")
fig.tight_layout()
plt.show()
"""
        ),
        md(
            """
### 11b — `n_chunks` distribution per scenario

How many chunks each chunker emits per doc. A chunker that produces hundreds
of chunks per doc is suspect (likely splitting on every sentence regardless
of size); a chunker that produces ≤1 chunks for a 4096-token target on
docs ≥ 4096 tokens is suspect (chunk threshold not firing).
"""
        ),
        code(
            """
chunk_summary = (
    scores.groupby(["scenario_id", "scenario_display", "chunker", "chunk_tokens"], dropna=False)
    ["n_chunks"].agg(["min", "median", "max", "mean"]).reset_index()
)
chunk_summary["chunk_tokens"] = chunk_summary["chunk_tokens"].astype("Int64")
chunk_summary
"""
        ),
        code(
            """
chunked_only = scores[scores.scenario_id != "S0"].dropna(subset=["chunk_tokens"]).copy()
chunked_only["chunk_tokens"] = chunked_only["chunk_tokens"].astype(int)
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.boxplot(
    data=chunked_only,
    x="scenario_display",
    y="n_chunks",
    order=[s for s in scenario_display_order if scores[scores.scenario_display == s].scenario_id.iloc[0] != "S0"],
    hue="chunker",
    hue_order=[c for c in CHUNKER_ORDER if c != "truncate"],
    palette={k: v for k, v in chunker_palette.items() if k != "truncate"},
    ax=ax,
    dodge=False,
    showfliers=False,
    legend=False,
)
ax.set_yscale("log")
ax.set_xlabel("scenario")
ax.set_ylabel("n_chunks per doc (log)")
ax.set_title(f"chunk-count distribution — study {study_cfg.study.name}")
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_horizontalalignment("right")
fig.tight_layout()
plt.show()
"""
        ),
        md(
            """
### 11c — Token analysis

Three diagnostics derived from the new `n_tokens` / `n_tokens_per_chunk` per-
record metadata: empirical chars-per-token recalibration (closes O5),
truncation loss on the S0 baseline, and chunk-size distribution per chunker.
"""
        ),
        md(
            """
#### 11c-i — `chars_per_token` calibration (closes O5)

Compares the per-language `chars_per_token` constants the corpus filter uses
(`configs/research/base.yaml`) against the empirical ratio measured on every
embedded doc. The corpus filter converts `min_tokens` → char threshold via
this ratio (`corpus_select.SelectionConfig.char_threshold`); a stale value
silently mis-selects boundary docs. If the empirical ratio drifts > ~5 %,
edit the YAML and re-run step 1.
"""
        ),
        code(
            """
# One row per doc — drop duplicate (scenario, ci_id) pairs so a doc that
# appears in 16 scenarios doesn't 16x-count its own len_chars / n_tokens.
unique_docs = (
    scores.drop_duplicates(subset=["ci_id"])
    .dropna(subset=["len_chars", "n_tokens"])
    .copy()
)
unique_docs = unique_docs[unique_docs["n_tokens"] > 0]
unique_docs["cpt"] = unique_docs["len_chars"].astype(float) / unique_docs["n_tokens"].astype(float)

# Sum-of-sums ratio: equivalent to a length-weighted mean — the right number
# for the corpus filter (which converts a token threshold into a char cutoff
# applied to the same total-length field).
empirical_weighted = (
    unique_docs.groupby("lg")
    .apply(lambda g: g.len_chars.sum() / g.n_tokens.sum(), include_groups=False)
)
yaml_defaults = study_cfg.corpus.chars_per_token
calib = pd.DataFrame(
    {"yaml_default": yaml_defaults, "empirical_weighted": empirical_weighted.round(3)}
)
calib["drift_pct"] = (
    (calib["empirical_weighted"] - calib["yaml_default"]) / calib["yaml_default"] * 100
).round(1)
print("chars_per_token — current YAML vs length-weighted empirical")
print(calib.to_string())
print()
big_drift = calib[calib["drift_pct"].abs() > 5.0]
if not big_drift.empty:
    print(f"!! drift > 5% on {sorted(big_drift.index)}; consider editing configs/research/base.yaml")
else:
    print("OK — all per-language drift within 5%; YAML defaults still valid.")
"""
        ),
        code(
            """
# Per-doc distribution: shows whether the ratio is stable or drifts within
# the corpus. A wide spread (e.g. p90/p10 > 1.4) means the YAML scalar is a
# poor proxy and the boundary-doc selection in corpus_select is noisy.
dist = (
    unique_docs.groupby("lg")["cpt"]
    .agg(["count", "mean", "std", "min",
          ("p10", lambda s: float(s.quantile(0.10))),
          ("p50", lambda s: float(s.median())),
          ("p90", lambda s: float(s.quantile(0.90))),
          "max"])
    .round(3)
)
print("chars_per_token — per-doc distribution")
print(dist.to_string())
"""
        ),
        code(
            """
# Visual: scatter of len_chars vs n_tokens per language. The slope of a
# line through the origin is the chars_per_token ratio; a tight band means
# the YAML scalar is a fine approximation, a fan-out means the ratio drifts
# with doc length (typically OCR-quality artefacts on long docs).
fig, axes = plt.subplots(
    nrows=1, ncols=len(LG_ORDER),
    figsize=(5.5 * len(LG_ORDER), 4.5),
    sharey=True,
)
if len(LG_ORDER) == 1:
    axes = [axes]
lg_colors = dict(zip(LG_ORDER, sns.color_palette("colorblind", n_colors=len(LG_ORDER)), strict=True))
for ax, lg in zip(axes, LG_ORDER, strict=True):
    sub = unique_docs[unique_docs.lg == lg]
    if sub.empty:
        ax.set_title(f"{lg} — (no docs)")
        continue
    ax.scatter(sub["n_tokens"], sub["len_chars"], s=14, alpha=0.55, color=lg_colors[lg])
    # Reference lines: YAML default + empirical-weighted, both through origin.
    xs = np.array([0, float(sub["n_tokens"].max())])
    yaml_ratio = float(yaml_defaults.get(lg, np.nan))
    emp_ratio = float(empirical_weighted.get(lg, np.nan))
    if not np.isnan(yaml_ratio):
        ax.plot(xs, xs * yaml_ratio, "--", color="gray",
                label=f"YAML {yaml_ratio:.2f}")
    if not np.isnan(emp_ratio):
        ax.plot(xs, xs * emp_ratio, "-", color="black", linewidth=1.2,
                label=f"empirical {emp_ratio:.2f}")
    ax.set_title(f"{lg}  n={len(sub)}")
    ax.set_xlabel("n_tokens")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
axes[0].set_ylabel("len_chars")
fig.suptitle(
    f"len_chars vs n_tokens — study {study_cfg.study.name}",
    y=1.02, fontsize=12,
)
fig.tight_layout()
plt.show()
"""
        ),
        md(
            """
#### 11c-ii — Truncation loss on the S0 baseline

`S0` truncates at `model.max_seq_length` (8190 for `gte-multilingual-base`).
For every gold doc with `n_tokens > 8190`, S0 silently drops the tail. This
table reports per-language, per-token-bucket: how often it happened, mean
tokens dropped, and Recall@5 within the bucket so we can see whether the
truncated docs are also the docs S0 fails on.
"""
        ),
        code(
            """
trunc_table = ev.truncation_loss_table(scores, baseline_id="S0", limit=8190)
if trunc_table.empty:
    print("S0 not present in scores — truncation loss table is empty.")
else:
    show = trunc_table.copy()
    show["pct_truncated"] = (show["pct_truncated"] * 100).round(1)
    show["mean_truncated_tokens"] = show["mean_truncated_tokens"].round(0).astype("Int64")
    show["recall_at_5"] = show["recall_at_5"].round(3)
    show["mrr"] = show["mrr"].round(3)
    print("S0 truncation loss — per (lg, token_bucket)")
    print(show.to_string(index=False))
"""
        ),
        md(
            """
#### 11c-iii — Chunk-size distribution & padding overhead

For every chunked scenario (S0 excluded), explodes `n_tokens_per_chunk` into
one row per chunk and reports: violin plot of actual chunk size, plus a
summary table with `stub_count` (chunks under 256 tokens — the pathology
where the trailing fragment drags the mean off-course), `padding_pct`
(fraction of the target window left empty on average), and `cv` (coefficient
of variation; high cv = uneven chunker — sentence-aware chunkers tend to
spike here on long-headline OCR).
"""
        ),
        code(
            """
chunk_rows = []
for sid, pool in inputs.pools.items():
    if pool.scenario.chunker_name is None:
        continue
    target = pool.scenario.chunk_tokens
    for per_chunk in pool.n_tokens_per_chunk:
        for tok in per_chunk:
            chunk_rows.append({
                "scenario_id": sid,
                "scenario_display": SCENARIO_LABELS[sid],
                "chunker": pool.scenario.chunker_name,
                "chunk_tokens_target": target,
                "actual": int(tok),
            })
chunk_df = pd.DataFrame.from_records(chunk_rows)

if chunk_df.empty:
    print("No chunked scenarios in this study (all scenarios are S0?).")
else:
    summary = (
        chunk_df.groupby(["scenario_id", "scenario_display", "chunker", "chunk_tokens_target"])
        ["actual"]
        .agg(["count", "mean", "std", "min", "max",
              ("stub_count", lambda s: int((s < 256).sum()))])
        .reset_index()
    )
    summary["padding_pct"] = (
        (summary["chunk_tokens_target"] - summary["mean"])
        / summary["chunk_tokens_target"] * 100
    ).round(1)
    summary["cv"] = (summary["std"] / summary["mean"]).round(3)
    print("chunk-size distribution per scenario")
    print(summary[[
        "scenario_id", "chunker", "chunk_tokens_target", "count",
        "mean", "stub_count", "padding_pct", "cv"
    ]].round(1).to_string(index=False))
    print()

    chunked_order = [
        SCENARIO_LABELS[s] for s in SCENARIO_ORDER
        if s in chunk_df["scenario_id"].unique()
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.violinplot(
        data=chunk_df,
        x="scenario_display",
        y="actual",
        order=chunked_order,
        hue="chunker",
        hue_order=[c for c in CHUNKER_ORDER if c != "truncate"],
        palette={k: v for k, v in chunker_palette.items() if k != "truncate"},
        ax=ax,
        cut=0,
        density_norm="width",
        inner="quartile",
        legend=False,
    )
    ax.set_xlabel("scenario")
    ax.set_ylabel("actual chunk size (tokens)")
    ax.set_title(f"chunk-size distribution — study {study_cfg.study.name}")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    fig.tight_layout()
    plt.show()
"""
        ),
        md(
            """
## 12 — Verdict

> **Edit this cell after running.** Three lines max per language. Cite the
> Δ Recall@5 from section 4 plus the supporting plot, and call out anything
> visible-but-not-explained-by-the-headline (e.g. position-bucket asymmetry
> a winner doesn't fix).

### `fr`

- _(fill in after running)_ — best scenario by Δ Recall@5: …
- Significant CI lift over S0? …
- Open follow-up: …

### `de`

- _(fill in after running)_ — best scenario by Δ Recall@5: …
- Significant CI lift over S0? …
- Open follow-up: …

## 13 — Provenance

Every row of `scores.parquet` carries `study_name` + (implicitly via
the embedding shards) `study_config_sha`. The same fingerprint is
embedded in every upstream artefact (`corpus.jsonl.bz2`,
`queries.jsonl.bz2`, `queries-embedded.jsonl.bz2`, `S{i}.jsonl.bz2`)
so two runs of the same study with different YAML edits stay
distinguishable in S3.
"""
        ),
        code(
            """
print(f"study={study_cfg.study.name}")
print(f"config_sha={study_cfg.config_sha}")
print(f"S3 prefix=s3://{study_cfg.s3.bucket}/{study_cfg.study_s3_root()}/")
print(f"local mirror={study_cfg.study_local_root()}")
print(f"scores parquet={parquet_path}")
"""
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "pygments_lexer": "ipython3",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        }
    )
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate notebooks/<study>-eval.ipynb."
    )
    parser.add_argument(
        "study",
        nargs="?",
        default="study-A-fit",
        help="study slug (default: study-A-fit). Must match a YAML in configs/research/.",
    )
    args = parser.parse_args(argv)

    cfg_path = CONFIGS_DIR / f"{args.study}.yaml"
    if not cfg_path.exists():
        print(f"error: {cfg_path} does not exist", file=sys.stderr)
        return 2

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(args.study)
    out = NOTEBOOKS_DIR / f"{args.study}-eval.ipynb"
    nbformat.write(nb, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
