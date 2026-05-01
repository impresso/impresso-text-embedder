# eval-harness — design notes

Step 8 of `research/chunking-eval`: turn the per-scenario doc
embeddings (from `embed_sweep`) and the per-query embedding shard
(from `query_embed`) into a per-(query, scenario) score table plus
the plots that answer the headline research question.

The deliverable is **one notebook per study** (`notebooks/<study>-eval.ipynb`)
backed by a shared, unit-tested helper module
(`src/impresso_text_embedder/research/eval.py`). The notebook
loads inputs from S3 via the helper, runs sanity checks, scores
every query against every scenario, and renders the verdict
table + plots inline. Plots stay in the rendered notebook
(checked into git so the report is readable without re-running);
no separate PNG dump.

## TL;DR

- **Form**: Jupyter notebook per study + a `research/eval.py` helper.
  No CLI script. The helper is the testable surface; the notebook
  is the analysis surface.
- **Inputs**: read straight from `s3://<bucket>/chunking-eval/<study>/`
  using a thin `ensure_local(study_cfg, filename)` helper that caches
  to the study's local mirror. Re-running cells doesn't re-download.
- **Metrics**: doc-level `Recall@1/5/10`, MRR; per-cell bootstrap CIs
  (B=1000); paired-bootstrap Δ vs the S0 truncate baseline. Per-language
  retrieval pool only (fr queries vs fr docs). Chunk-level metrics
  (IoU / Precision_Ω) deferred — see open items.
- **Plots**: Recall@5 by scenario (faceted by lg), Recall@5 vs
  `chunk_tokens` (log₂ x-axis, faceted by lg), position-bucket
  robustness, MRR comparison, query-type ablation, rank-of-gold
  boxplot, `n_chunks` distribution. Seaborn theme = colorblind palette,
  white-grid context, S0 baseline overlaid as a dashed reference line
  on every primary plot.
- **No production CLI changes.** This branch is research-only;
  findings ship as a recommendation in the per-study verdict cell.

## Inputs / outputs

### Inputs (read from S3, cached locally)

- `s3://{bucket}/{study.s3_root}/queries-embedded.jsonl.bz2` —
  one record per query: `{query_id, ci_id, embedding, lg,
  query_type, position_bucket, position_chars, references,
  query_text, model_id, ts, study_name, study_config_sha}`.
  Source: `research.query_embed`.
- `s3://{bucket}/{study.s3_root}/{scenario_id}.jsonl.bz2` for
  every scenario in the registry — one record per doc:
  `{ci_id, embedding, lg, year, provider, alias, ocrqa,
  len_chars, n_chunks, scenario_id, chunker, chunk_tokens,
  model_id, ts, ci_type, study_name, study_config_sha}`.
  Source: `research.embed_sweep`.

### Outputs (local; not uploaded)

- `tmp/chunking-eval/<study>/eval/scores.parquet` — one row per
  `(query_id, scenario_id)` with rank-of-gold, Recall@k flags,
  MRR contribution, plus all stratification keys
  (`scenario_id, scenario_label, chunker, chunk_tokens, lg,
  query_type, position_bucket, n_chunks, pool_size`). The
  parquet is the durable evidence; the notebook's plots are
  derived from it.
- The notebook itself, executed and committed, with rendered
  plots inline. **The notebook IS the report.** No separate
  HTML / PDF export step at v1.

## Mechanism

1. **`ensure_local(study_cfg, filename)`** — download
   `s3://{bucket}/{study_cfg.s3_key(filename)}` to
   `study_cfg.local_path(filename)`, reusing the file if it
   already exists. The production `staged_input` deletes its
   tempfile on context exit, which is exactly wrong for a
   notebook that re-runs cells; a stable local cache is the
   right primitive here.
2. **`load_eval_inputs(study_cfg)`** — pull queries +
   every scenario shard, return a frozen `EvalInputs` (queries
   tuple + scenario-id-keyed `ScenarioPool` dict + the registry's
   scenario list). Per-scenario rows are stored as a flat
   `(N, dim)` numpy matrix plus parallel `ci_ids` / `langs` /
   `n_chunks` arrays — fast to slice by language at scoring time.
3. **`run_sanity(inputs)`** — collect (not raise) issues:
   inconsistent embedding dim across artefacts, scenarios
   covering different `ci_id` sets, queries whose gold `ci_id`
   is absent from the doc pool, non-finite embeddings, non-unit
   norms (tolerance 1e-2 to absorb the bf16 + 5-dp-rounding
   round-trip). The notebook prints the report as a styled
   pandas DataFrame and asserts `report.passed` before scoring.
4. **`score_queries(inputs)`** — for each `(query, scenario)`,
   slice the pool to the query's language, compute
   `cosine_scores(query.embedding, sub_pool)`,
   `rank_of_gold(scores, gold_idx)`, and the Recall@k flags.
   Returns a tidy pandas DataFrame. Per-language pool is
   memoised once per `(scenario_id, lg)` pair so the inner
   loop is a single matrix-vector product per query.
5. **`baseline_delta_table(scores, baseline_id="S0")`** —
   pivot scores by `(query_id, scenario_id)`, paired-bootstrap
   the per-query Δ vs the baseline column, return a DataFrame
   sorted by Δ descending with `beats_baseline` /
   `loses_to_baseline` flags (`low > 0` and `high < 0`
   respectively at the chosen CI). The verdict source.
6. **Plots (in the notebook, not the helper)** — every
   primary plot facets on `lg`, hues on `chunker` family with
   a colorblind palette, draws the S0 mean as a dashed gray
   reference line, and uses seaborn's built-in
   `errorbar=("ci", 95)` (which bootstraps internally) so we
   don't double-bootstrap. The `chunk_tokens` axis on the
   convergence plot uses `set_xscale("log", base=2)`. The
   rank-of-gold boxplot uses `symlog` y-scale because most
   ranks are near 1 but the upper whiskers can reach the
   pool size.

### Tie-breaking on `rank_of_gold`

Pessimistic (competition-rank): an item tied with k other items
that score *higher or equal* gets rank `k+1`. Prevents a
degenerate scenario that produces many near-zero embeddings
from inflating its Recall@1 by tying with a flat-zero pool.
Tested in `tests/test_research_eval.py::test_rank_of_gold_breaks_ties_pessimistically`.

### Per-language pool — explicit

Every query ranks against the same-language slice of the
scenario's doc pool (`pool[pool.lg == query.lg]`). Cross-lingual
retrieval would mix the natural in-language similarity signal
with a weaker cross-lingual one and confound the chunker
comparison; it is the gated O4 ablation for after the
monolingual sweep produces a clear winner. `pool_size` is
emitted on every score row so a downstream eye can spot any
cell that fell to zero.

## Rejected alternatives

- **CLI-only — no notebook.** The CLIs already exist for the
  pipeline upstream; another CLI to dump scores parquet would
  be the natural pattern. Rejected on the user's directive
  ("Only notebook for the moment, one notebook per study"). The
  benefit of the notebook is that the report and the evidence
  are the same artefact: the inline plots ARE the deliverable
  and the Δ-table ARE the verdict. A CLI would force a separate
  reporting step.
- **Notebook with everything inline (no helper).** Tempting on a
  one-off, costly long-term: every metric needs a unit test
  (CLAUDE.md → "Testing"), and tests live next to importable
  modules. Pulling the metrics + loaders + sanity into
  `research/eval.py` keeps each function independently testable
  while the notebook stays an analysis artefact. Mirrors the
  shape of the upstream research module pattern (one
  module per CLI, peer test file).
- **Plots saved to disk + a thin notebook that just embeds
  PNGs.** Rejected — the user's "no local plot only" directive,
  and the rendered ipynb already inlines plots when committed.
  Saving to disk would duplicate state.
- **`papermill` / `jupytext` source-of-truth in a `.py` file.**
  Considered — would let a single CLI regenerate the notebook
  per study with a `STUDY_CONFIG_PATH` parameter override.
  Rejected at v1 — adds a dependency and a build step for
  the simple case of "one study". The generator script
  (`scripts/build_eval_notebook.py`) already gives us
  per-study regeneration via `nbformat.v4.new_notebook` without
  a third-party tool. Promote to `papermill` only if we end up
  parameter-sweeping the notebook itself (different `k_values`,
  different baseline scenarios, etc.).
- **`token-level Recall@k` (Chroma's primary metric) at v1.**
  Rejected — Chroma's metric is designed for **chunk-level
  retrieval**: given a query, retrieve the top-k chunks across
  the corpus, ask what fraction of the gold-reference tokens
  are covered. We retrieve **whole docs** (`--embedding-level
  text`); the natural metric is the binary "did the gold doc
  land in top-k". Switching to token-level Recall would require
  chunk-level embeddings, which are out of scope on this
  branch. Promote only if `--embedding-level chunk` lands as
  a follow-up branch.
- **NDCG@k.** Rejected — relevance is binary on this corpus
  (gold doc / not gold doc), and binary-relevance NDCG@k
  collapses to a rank-weighted Recall that MRR already covers.
  Adding NDCG would dilute the verdict table without adding
  signal.
- **Pre-computing CIs via `ev.bootstrap_ci` and passing them
  to `seaborn.barplot` via `xerr`/`yerr`.** Rejected — seaborn
  v0.12+ does its own bootstrap inside `barplot(errorbar=("ci",
  95))` and the result is identical for our use case (mean of
  binary `recall_at_5`). Pre-computing would force a
  manual-matplotlib path on every plot for no methodological
  gain. The standalone `ev.bootstrap_ci` is still useful for
  the Δ-table (paired) and for any custom plot the notebook
  grows later.
- **Chunk-level IoU / Precision_Ω at v1.** Rejected on scope —
  these need chunker span recovery (start/end char offsets per
  chunk), which doesn't exist on this branch. The fixed-window
  chunker would need `tokenizer(..., return_offsets_mapping=True)`
  bookkeeping; `token-budget` would need sentence-offset
  reconstruction; chonkie exposes `Chunk.start`/`Chunk.end`
  natively but it's a separate code path. The doc-level
  metrics already answer the headline research question (does
  forced sub-context chunking change doc-level embedding
  quality?). Chunk-level metrics are diagnostic, not
  decisional, on this branch. Listed as O15 below.
- **One notebook covering all studies, parameterised.**
  Rejected — the user explicitly asked for one per study. Also
  cleaner: each study's verdict cell tells a different story
  (study-A-fit asks "does chunking help when the doc fits?"
  while study-B-overflow asks "where does truncate cliff?"),
  and inlining both into one notebook would obscure the two
  separate questions.
- **Score every query against the union of all-language doc
  pools.** Rejected — fr queries against de docs would test
  cross-lingual retrieval, a different (gated O4) question.
  Per-language scoring keeps the answer specific.

## Open items

- **O15 — chunk-level IoU / Precision_Ω.** The diagnostic
  metric for chunker boundary quality. Requires a
  `chunk_with_spans()` helper that returns `(text, char_start,
  char_end)` triples per chunk under a given scenario. Easiest
  on `fixed-window` (tokenizer offset_mapping), tractable on
  `token-budget` (sentence offsets), trivial on `semantic`
  (chonkie exposes spans). Promote to a follow-up step once
  the doc-level verdict lands and we know which scenarios
  warrant the deeper diagnostic.
- **O16 — verdict cell automation.** The notebook's "edit me
  after running" cell is currently a stub. Could be auto-
  populated from the Δ-table top row via a small templating
  helper. Trivial; gated on the first real run of study-A-fit
  surfacing what shape the verdict actually wants.
- **O17 — multi-study comparative notebook.** Once both
  study-A-fit and study-B-overflow have run, a `cross-study.ipynb`
  could load both `scores.parquet` files and answer "does the
  best chunker on A-fit also win on B-overflow?". Not a
  per-study artefact; would live next to the per-study
  notebooks under `notebooks/cross-study-eval.ipynb`. Listed
  as a follow-up step, not included here.
- **O18 — cross-lingual ablation (was O4).** Run the eval with
  `pool[pool.lg != query.lg]` and compare. Gated on a
  monolingual winner; if the headline says "S0 truncate is
  fine on study-A-fit" then the cross-lingual question
  becomes "do chunkers help cross-lingual retrieval even when
  they don't help in-language?" — a different research project.
- **O19 — corpus-design changes for study-A-fit re-run** (raised
  by the 2026-05-01 post-mortem below). Three coupled changes,
  ordered by leverage: (a) prefilter the corpus to a
  topic-bounded slice so retrieval competition is real — done
  via Impresso topic clusters at sample time, used as a
  *sampling filter only* (relevance signal stays Chroma-style);
  (b) bump `corpus.n_per_lg` 200 → 300 to bring fr CI
  half-width on Δ Recall@5 from ±0.025 down to ~±0.020 (the fr
  signal is currently borderline at 2-4 pp deltas); (c) stratify
  sampling by decade so fr (currently 1880s-1900s heavy) and de
  (currently 1940s-1970s heavy) cover the same era band — closes
  O6 from corpus-selection. Without (a) the eval is dominated by
  ceiling queries (~57% at S0 rank=1); without (c) any fr-vs-de
  comparison is partly an OCR-era artefact. **Keep both query
  types** (question + topical-phrase): per-scenario rankings only
  correlate at pearson 0.66 on fr (0.97 on de), so they measure
  meaningfully different things. If LLM budget is a constraint,
  lift `queries_per_bucket` 1→2 on `topical-phrase` only — it's
  the harder + more realistic eval (S0 Recall@5: question 0.84
  vs topical-phrase 0.72 on fr).
- **O20 — alternative metrics: cosine margin + softmax NLL**
  (raised by the 2026-05-01 post-mortem below). Add two
  per-query columns to `score_queries` output, computed from the
  cosine matrix already built:
  ```python
  margin = cos(q, gold) - max_{d != gold} cos(q, d)
  nll    = -log softmax(cos(q, .) / 0.05)[gold]
  ```
  Both are continuous, don't saturate at rank=1, and surface
  embedding-confidence signal that Recall@5 hides. Empirically
  (verified on study-A-fit data): margin Δ vs S0 ranking
  correlates with Recall@5 Δ at spearman 0.63 fr / 0.80 de — so
  the metrics are *additional*, not redundant. Concrete win
  surfaced: S9 `semantic-512` shows Recall@5 Δ +0.064 on de
  (looks like a moderate win) but NLL Δ +0.45 nats (much worse
  embedding confidence) — the chunker is occasionally winning
  rank-1 lottery while degrading the embedding everywhere else.
  Recall@5 cannot see this. Implementation is ~10 LoC inside
  `eval.score_queries` and a mirror of the headline Δ-table /
  per-scenario plot in the notebook for each new metric; keep
  Recall@5 alongside as the legacy contract. **The
  three-metric agreement (or disagreement) is the actual
  finding**, not any one in isolation. τ=0.05 chosen to match
  the contrastive-training regime for unit-norm cosines; expose
  as a kwarg if needed for ablation.
- **O21 — hard-negative-restricted Recall@k** (raised by the
  2026-05-01 post-mortem below; complements O19 (a)). Post-hoc
  topic-bounding equivalent that needs zero corpus-construction
  changes: for each query, pre-mine top-K=10 hardest non-gold
  docs from the *full* pool using the S0 baseline cosines,
  freeze that set, then re-evaluate every scenario as a
  (1+K)-element classification (gold vs the same K hard
  negatives across scenarios — keeps the eval fair). Closes the
  ceiling-effect dilution that wastes ~50% of queries on the
  current pool. Mirrors what the Chroma eval does at corpus-
  construction time (cosine-filter queries whose excerpt
  similarity is below threshold) but applied at scoring time so
  no LLM re-run is needed. Land alongside O20 — same `eval.py`
  refactor. Cross-validate with O19 (a): if the topic-bounded
  corpus and the hard-negative subset agree on the headline,
  they reinforce each other; if they disagree, the disagreement
  is itself the finding.
- **O22 — promote O15 (token-level IoU / Precision_Ω) sooner**
  (raised by the 2026-05-01 post-mortem below). Originally
  framed as a deferred deeper-diagnostic; the post-mortem
  reframes it as the *principled* metric for chunking eval
  specifically, since it's pool-independent — the topic-
  orthogonality problem disappears at the root. Worth promoting
  to a follow-up step rather than waiting for the doc-level
  verdict to land, because O20/O21 only mitigate the symptom
  (saturation) while O15 attacks the cause (chunker quality is
  not the same question as doc-level retrieval quality).
  Direct cite: Chroma Research, "Evaluating Chunking
  Strategies for Retrieval".

## Findings from study-A-fit eval (2026-05-01)

Post-mortem of the first study-A-fit run before re-running with a
calibrated `chars_per_token` (closes O5). The scores parquet
(notebooks/tmp/chunking-eval/A-fit/eval/scores.parquet,
config_sha at run-time) and the per-scenario embedding shards
were the substrate. The four findings below drive open items
O19-O22; capture them here so the next study cycle starts from
the diagnosis, not from the headline plot.

### Finding 1 — N=200 articles is adequate on de, borderline on fr

200 articles/lg → ~1100 queries/lg after verification loss →
95% paired-bootstrap CI half-width on Δ Recall@5 ≈ ±0.025.
Observed deltas: de best +0.10 (S1) — 9/12 chunked scenarios
significantly differ from S0; fr best +0.04 (S7) — only 5/12
significantly differ. The fr signal sits at the resolution
limit. Empirical std of paired diffs implies n≈340 articles
needed to hit ±0.020 half-width and resolve fr 2 pp differences;
n≈1300 needed for ±0.010 (not worth the LLM cost). Drives O19 (b).

### Finding 2 — Both query types are needed; topical-phrase is the harder/more realistic one

Per-scenario Recall@5 rankings from question-only vs
topical-phrase-only correlate at pearson 0.66 on fr (top-3
disagreement: question {S2,S6,S1} vs topical {S7,S6,S2}) and
0.97 on de. So on fr they measure *different things* and
collapsing to one type loses signal. Difficulty asymmetry: S0
Recall@5 is 0.842 (question) vs 0.724 (topical-phrase) on fr,
0.799 vs 0.744 on de — topical-phrase is consistently harder
and looks more like real archive-search behaviour (e.g. "projets
scolaires 1954" vs "Quelle île a été le lieu de la découverte
du corps d'Andrée…?"). Position buckets earn their keep too:
fr question rank=1 fraction is 60% head / 60% mid / 75% tail
(tail-loaded — last paragraphs in 19th-century fr articles
carry distinctive signatures), de question is 75% / 55% / 47%
(head-loaded — masthead/title vocabulary dominates). Drives the
"keep both query types" guidance in O19; does **not** support
collapsing buckets.

### Finding 3 — Random-topic 200-article sample → ~50% ceiling queries

S0 baseline rank distribution (pool=200): rank=1 fraction ≈ 57%
on de, 60% on fr; rank≤5 ≈ 75-77%. So ~half of all queries
**cannot move under any chunker** — they dilute the headline.
Restricted to "hard" queries (S0 rank > 5; n=257 fr, n=290 de)
the chunker deltas explode 6-15×: fr best Δ Recall@5 +0.52 (S6),
de best +0.60 (S1). The observed 2-10 pp headline is the
diluted version of a 50-60 pp underlying effect. Random-topic
sampling lets the embedder discriminate articles by period
vocabulary / OCR artefacts / unique entities alone. Drives O19
(a) (topic-bounded corpus) and O21 (hard-negative subset eval as
a free post-hoc equivalent).

Compounding this: the first-run corpus has severe
provider/decade skew — de is 91% post-1940 (DTT-dominated, n=134),
fr is 53% pre-1900. fr-vs-de delta differences (3 pp vs 10 pp)
may be partly an era artefact, not a language-intrinsic
difference. Drives O19 (c).

### Finding 4 — Recall@5 hides embedding-confidence signal

Two metrics computed on the existing eval (margin =
cos(q,gold)−max non-gold cosine; NLL = −log softmax(cos/τ)
with τ=0.05) extract additional, actionable signal:

- On the ~50% of queries already at S0 rank=1 ("ceiling"
  queries), **every chunked scenario has smaller margin than
  S0** (de: −0.016 to −0.063; fr: −0.008 to −0.041). Chunking
  is trading confidence on easy queries for occasional rank-1
  wins on hard queries — this is invisible under Recall@k.
- The semantic family (S9-S12) looks competitive on Recall@5
  (de: 0.81 at S9-S11) but is much worse on NLL (de: +0.45 to
  +0.98 nats vs S0). The chunker is producing less confident
  embeddings even when it sometimes ranks gold higher.
- Spearman rank correlation between margin-ordering and
  Recall@5-ordering is 0.63 on fr / 0.80 on de — so margin and
  NLL give genuinely additional dimensions.

Drives O20 and O21. Both metrics are ~10 LoC additions to
`eval.score_queries` reusing the cosine matrix already built;
zero new infra. Independent of corpus construction so they
land in parallel with the O19 corpus changes.

### Cited literature

The metric directions in O20-O22 are not novel; they are the
standard alternatives the IR community has been using for
years. Anchors for the curious:

- Chroma Research, "Evaluating Chunking Strategies for
  Retrieval" — token-level Recall, Precision, Precision_Ω, IoU.
  https://research.trychroma.com/evaluating-chunking
- Bruch et al., "An Analysis of the Softmax Cross Entropy Loss
  for Learning-to-Rank with Binary Relevance" (SIGIR 2019) —
  formal connection of softmax NLL to MRR/NDCG.
  https://research.google/pubs/an-analysis-of-the-softmax-cross-entropy-loss-for-learning-to-rank-with-binary-relevance/
- Wang & Liu, "Negative Margin Matters: Understanding Margin in
  Few-shot Classification" (ECCV 2020) — score-margin as a
  continuous discriminative metric.
  https://arxiv.org/abs/2003.12060
- Wang & Isola, "Understanding Contrastive Representation
  Learning through Alignment and Uniformity on the Hypersphere"
  (ICML 2020) — pool-free embedding-quality diagnostics if
  O20/O21 give muddy answers.
  https://arxiv.org/abs/2005.10242
- Moreira et al., "NV-Retriever: Improving text embedding
  models with effective hard-negative mining" (2024) —
  TopK-MarginPos / TopK-PercPos for false-negative-aware mining;
  reach for if O21's static top-K turns out to mine paraphrases.
  https://arxiv.org/abs/2407.15831

## Reproducing

Build a per-study notebook from the shared template:

```bash
uv run python scripts/build_eval_notebook.py study-A-fit
uv run python scripts/build_eval_notebook.py study-B-overflow
```

Run a notebook end-to-end:

```bash
uv sync --extra research --extra dev
uv run jupyter lab notebooks/study-A-fit-eval.ipynb
# Or non-interactive:
uv run jupyter execute notebooks/study-A-fit-eval.ipynb
```

The notebook will pull every artefact for the study from
S3 (cached under `tmp/chunking-eval/<study>/`) on first
run. To force a re-download, delete the local cache or pass
`force=True` to the loader cell.

## Upstream references

- `research/embed_sweep.py` — produces the per-scenario
  doc-embedding shards consumed here.
- `research/query_embed.py` — produces the per-query embedding
  shard consumed here.
- `research/scenario_builder.py` — `ScenarioRegistry` is the
  source of truth for scenario order on every plot's x-axis.
- `tests/test_research_eval.py` — synthetic-fixture round-trip
  for the loaders + metrics + scoring pipeline.
- Chroma Research, "Evaluating Chunking Strategies for
  Retrieval" (Jul 2024) —
  https://research.trychroma.com/evaluating-chunking — methodology
  cite for the LLM query-generation protocol; we depart from it
  in retrieval granularity (doc-level, not chunk-level) and
  metric definition (binary Recall, not token-level).
- Liu et al., "Lost in the Middle" (TACL 2024) —
  https://arxiv.org/abs/2307.03172 — head/mid/tail bucketing
  rationale.
