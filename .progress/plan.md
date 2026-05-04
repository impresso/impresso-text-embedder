# research/chunking-eval — plan

Live ledger for the chunking-evaluation research on this branch. The migration
that produced the production embedder is **done** and lives in
[`../.history/plan.md`](../.history/plan.md); the locked decisions it recorded
are the invariants this research operates within (see CLAUDE.md → "Decisions
inherited from the migration").

## Research question

Production currently chunks only when `tokens(doc) > model_max_tokens` (8192
for `gte-multilingual-base`). The argument was theoretical: CLS-pooling means
chunk-and-pool drops cross-chunk attention without a recovery mechanism. The
question this branch answers empirically: **on Impresso historical newspapers,
does forcing sub-8k chunking + mean+L2 pool produce better doc-level
embeddings than one-shot encode at 8190?**

Answer is per-language (fr / de / lb) — chars-per-token differs across
Romance / Germanic / Luxembourgish, so the optimum may differ.

## Scope (locked with the user)

- **Embedding level**: `--embedding-level text` only. One vector per doc,
  mean+L2 when chunked. Chunk-level retrieval is a different question and is
  out of scope.
- **Languages**: fr / de / lb. Italian deferred.
- **Relevance signal**: Chroma-style synthetic LLM query+excerpt pairs only.
  Native Impresso annotations (topic clusters, text-reuse, NER) are **not**
  used as a relevance signal — keeps methodology comparable to the published
  chunking-eval literature and avoids inventing a bespoke signal.
- **Strategies**: `truncate` at 8190 (production baseline); `fixed-window`
  and `token-budget` at {512, 1024, 2048, 4096, 8190}; `semantic` (chonkie) at
  one calibrated size.
- **Aggregation**: `mean` for the primary sweep. Length-weighted / max /
  first-chunk ablations only on the winning chunker.
- **Corpus**: long articles (>4000 raw tokens), high OCR quality, topic-
  bounded slice only as far as needed to keep generation cost bounded.

## Status at a glance

| # | Slug | Status |
|---|------|--------|
| 1 | [corpus-selection](./corpus-selection/notes.md) | done |
| 2 | [corpus-fetch](./corpus-fetch/notes.md) | done |
| 3 | [embedding-sweep](./embedding-sweep/notes.md) | done |
| 4 | [query-generation](./query-generation/notes.md) | wip |
| 5 | [study-config](./study-config/notes.md) | done |
| 6 | study-A-fit + study-B-overflow runs | todo |
| 7 | [query-embed](./query-embed/notes.md) | wip |
| 8 | [eval-harness](./eval-harness/notes.md) | wip |
| 9 | [semantic-chunker-fixes](./semantic-chunker-fixes/notes.md) | done |
| 10 | [aggregation-sweep](./aggregation-sweep/notes.md) | done |

## Steps

### 1 — corpus-selection

Per-language manifest of long, high-OCR newspaper articles for the
chunking-strategy sweep. Streams the 30 GB langident-aggregated jsonl in a
single pass, filters to `tp=article` + `ocrqa>=0.9` + `len >= min_tokens *
chars_per_token[lg]` + curated providers + year window, and per-language
samples N entries deterministically. Output: 200 fr + 200 de articles
manifest at `tmp/chunking-eval/corpus-manifest.jsonl`. lb dropped from the
default sweep — corpus property: long lb articles are systematically low-OCR
(every `len>=14000` lb article in the corpus has `ocrqa<=0.67`). Provider
defaults were corrected after the first run revealed NZZ/SWA/SUB carry
almost no de articles (real carriers are SNL, FedGaz, BNL). Code at
`src/impresso_text_embedder/research/corpus_select.py`; design narrative,
rejected alternatives, and frozen Q1–Q7 decisions in
[`./corpus-selection/notes.md`](./corpus-selection/notes.md).

### 2 — corpus-fetch

Materialises the manifest into a single `.jsonl.bz2` corpus shard the
downstream sweeps consume. Groups manifest entries by their rebuilt
source file (400 entries → 222 unique yearly shards), downloads each
shard once via the production multipart-parallel transfer config, scans
it locally with an early break once every wanted ci_id has been picked
up, falls back to `text.rebuild_ft_from_offsets` when a rebuilt record
lacks a precomputed `ft`, and writes records back **in manifest order**
so the downstream sweep sees a deterministic input. Output schema:
`{ci_id, lg, year, provider, alias, len_chars, ocrqa, tp, ft, sents,
lingproc_path?}` — manifest fields plus the rebuilt payload, with
`sents` preserved for future sentence-aware chunkers. Land path:
`s3://140-processed-data-sandbox/chunking-eval/corpus/corpus-v1.jsonl.bz2`
(research outputs are segregated under the `chunking-eval/` prefix —
they never write to the production `embeddings/docs/...` convention).
Code at `src/impresso_text_embedder/research/corpus_fetch.py`; design
narrative, rejected alternatives, and worker-count calibration in
[`./corpus-fetch/notes.md`](./corpus-fetch/notes.md).

### 3 — embedding-sweep

Run every chunking strategy in scope over the corpus shard from step 2,
one Run:AI job per scenario, write one `.jsonl.bz2` of doc-level
embeddings per scenario. 16 scenarios live in
`research/scenarios.py`: `S0` truncate baseline plus three chunker
families — `fixed-window` (S1–S5), `token-budget` (S6–S10), `semantic`
chonkie (S11–S15) — each covering the *same* five chunk sizes
{512, 1024, 2048, 4096, 8190}. The grid is intentionally symmetric so
cross-family comparisons at a fixed size are a one-row lookup.
Aggregator is `mean` across the board (mean+L2 per the locked scope).
Each job's output lands at
`s3://140-processed-data-sandbox/chunking-eval/embeddings/<scenario_id>_<label>/corpus-v1.jsonl.bz2`.
Per-record output combines the production text-level schema
(`{ci_id, model_id, embedding, size, ts, ci_type}`) with manifest
metadata (`{lg, year, provider, alias, ocrqa, len_chars}`) and
sweep-level fields (`{n_chunks, scenario_id, chunker, chunk_tokens}`)
so downstream eval can stratify in one pass. The sweep CLI sets
`LongDocConfig.model_max_tokens = scenario.chunk_tokens` so a target of
e.g. 512 chunks every doc longer than 512 tokens — needed because the
production embedder's "chunk only when doc > model_max_tokens" guard
would otherwise skip every doc ≤ 8192 and defeat the experiment.
Scenarios that need a `stride` kwarg on `FixedWindowStrategy`
(sliding-window) or new aggregators (length-weighted/max/first-chunk)
are out of scope on this step — the registry only emits scenarios the
current chunking + aggregation registries can build. Code at
`src/impresso_text_embedder/research/{scenarios,embed_sweep}.py`;
Make targets `runai-submit-research SCENARIO=Sx` and
`runai-submit-research-all` (loops the registry). Design narrative,
rejected alternatives (production-schema field, side-channel callback,
single-job loop, per-language split), and the duplicate-chunker-pass
rationale for `n_chunks` in
[`./embedding-sweep/notes.md`](./embedding-sweep/notes.md).

### 4 — query-generation

Generate a small, position-stratified synthetic
`(query, gold_excerpts)` set against the corpus shard from step 2 so
the eval step (TBD) can score the 16 scenario embeddings from step 3
with token-level Recall@k / IoU / Precision_Ω. Methodology mirrors
[Chroma's chunking-eval protocol](https://research.trychroma.com/evaluating-chunking)
(LLM is shown the whole doc, asked for a query whose answer is
contained in the doc, plus verbatim references) with one explicit
modification: queries are **stratified into 3 position buckets**
(head / mid / tail of the source doc by char offset) so a chunker
can't win by silently dropping the back half of every article. Two
queries per (doc, bucket) — one `question` and one
`topical-phrase`, deterministically (no RNG) — so 400 docs × 3
buckets × 2 types ≈ ~2400 generations. LLM is
`Qwen/Qwen3-30B-A3B-Instruct-2507` via the EPFL RCP AIaaS endpoint
(`https://inference.rcp.epfl.ch/v1`, OpenAI-compatible); auth via
`RCP_API_KEY` from `.env`. Both query types get equal coverage at
every bucket so the eval can stratify by `(bucket, query_type)`
directly; summary-style queries are out (inflate Recall via lexical
overlap). The only quality gate at v1 is **verbatim-anchor
verification** — each LLM-emitted reference must appear as an exact
substring of the source `ft` *and* fall inside the position bucket;
queries with no surviving references are dropped. Cosine
relevance/dedup filtering is deferred to O7 (no retention-rate
baseline yet to tune against). Output lands at
`s3://140-processed-data-sandbox/chunking-eval/queries/queries-v1.jsonl.bz2`.
Code at `src/impresso_text_embedder/research/query_generate.py`;
Make target `research-query-generate`. Cost at Qwen3-30B-A3B prices
≈ $0.80 / ~30–45 min wallclock at the 5-parallel AIaaS rate. Design
narrative, rejected alternatives (OpenAI SDK, CaaS fallback,
quintile buckets, sentence-aligned subsampling, summary as a query
type, cosine filter at v1), and open items (cosine filter
calibration, stats sidecar, Qwen3 OCR-noise sanity check) in
[`./query-generation/notes.md`](./query-generation/notes.md).

### 5 — study-config

Refactored the four research CLIs to read all hyperparameters from a
single YAML config per *study*, replacing the per-module
`DEFAULT_*` constants + Make variables + CLI flags pattern that
allowed the chunk-grid-vs-corpus inconsistency surfaced in the
review of steps 1/3. New module
`src/impresso_text_embedder/research/study_config.py` ships the
Pydantic schema + `load_study_config(path)` loader with single-level
`extends:` deep-merge, `{study}` path templating validated at load
time, and `config_sha` provenance fingerprint. New module
`scenario_builder.py` replaces the hardcoded 16-row `_SCENARIOS`
tuple with `build_scenarios(cfg.scenarios)` auto-numbered `S0..SN`
plus a `ScenarioRegistry` lookup helper; the Makefile shells out to
`python -m research.scenario_builder --config <path> --list-ids` so
the build matrix stays in sync with the YAML. Layout under
`configs/research/`: `base.yaml` (cross-study invariants) +
`study-v1.yaml` (frozen pre-refactor defaults — 16-row grid
preserved exactly) + `study-A-fit.yaml` (docs ≤ 8192 tokens,
13-scenario grid) + `study-B-overflow.yaml` (docs ≥ 16384 tokens,
16-scenario grid). Each of the four CLIs gained `--config <path>`;
per-flag CLI args still override field-by-field. `embed_sweep` and
`query_generate` emit `study_name` + `study_config_sha` in every
output record. `corpus_select` gained a `max_tokens` upper bound
(was a hole in the schema). Makefile driven by `STUDY ?= study-v1`;
target the 16-scenario v1 sweep with `make
runai-submit-research-all`, study A with `make STUDY=study-A-fit
runai-submit-research-all`, etc. CLAUDE.md → "Decisions inherited
from the migration" gained a "Study-config YAML" entry listing the
load-time invariants. 417 tests pass. Code at
`src/impresso_text_embedder/research/{study_config.py,scenario_builder.py}`;
design narrative, rejected alternatives (single multi-study YAML,
TOML, Hydra, chained `extends:`, globally unique scenario ids,
auto-derived chunkers), and the deferred regression check (run
new pipeline against `study-v1.yaml` and diff manifest against the
existing `corpus-v1.jsonl.bz2` ci_ids) in
[`./study-config/notes.md`](./study-config/notes.md).

### 6 — study-A-fit + study-B-overflow runs

Run the four-step pipeline (corpus-select → corpus-fetch →
query-generate → embedding-sweep) against each of the two new
studies and emit a single comparative report answering the
chunk-grid-vs-corpus decomposition the previous turns surfaced.
Pre-conditions: extend the step-1 eligible-count scan to
thresholds `{8192, 12288, 16384, 20000}` per language to confirm
the n=200 pool size for study B is feasible (de may force a
threshold relaxation to 12288 or asymmetric N). Also blocked on
step 7 landing — the eval harness consumes the
`queries-embedded.jsonl.bz2` artefact this step does not
produce. Acceptance bar: two `.jsonl.bz2` shards landing at
`chunking-eval/A-fit/corpus.jsonl.bz2` and
`chunking-eval/B-overflow/corpus.jsonl.bz2`, plus the per-scenario
embedding sweeps under `chunking-eval/{study}/embeddings/`, plus
queries under `chunking-eval/{study}/queries.jsonl.bz2`. The eval
report (token-level Recall@k, IoU, Precision_Ω stratified by
`(study, scenario, chunker, chunk_tokens, position_bucket, lg)`) is
the deliverable of this step.

### 7 — query-embed

Materialise per-query embeddings to S3 once so the eval step in
step 6 can join against per-scenario doc embeddings without
re-loading the embedding model. New CLI
`src/impresso_text_embedder/research/query_embed.py` mirrors
`embed_sweep`'s scaffolding (study YAML resolution,
`staged_input`/`staged_output`, log dir under
`experiments/chunking-eval/<study>/`, `--no-upload` / `--limit`
smoke-test knobs, `_resolve_value(cli, cfg, fallback)` knob
precedence) and reuses the pinned model + revision declared in
the study's `embed:` block. Reads the queries shard from step 4
(`s3://{bucket}/{study.s3_root}/queries.jsonl.bz2`), encodes via
a single `model.encode_texts` call (no chunking, no aggregation,
no record filtering — queries are short plain strings, never
long-doc), and writes per-query records `{query_id, ci_id,
embedding, size, model_id, ts, lg, query_type, position_bucket,
position_chars, references, query_text, study_name,
study_config_sha}` to `queries-embedded.jsonl.bz2` under the
same study prefix. New constant `QUERIES_EMBEDDED_FILENAME`
lands in `research/study_config.py` next to `QUERIES_FILENAME`.
Make target `runai-submit-query-embed STUDY=<name>` mirrors
`runai-submit-research`'s shape; logs land at
`/rcp-scratch/<user>/experiments/chunking-eval/<study>/<YYYY-MM-DD>/query-embed.log`.
Step 6's eval harness is the consumer.

Code at `src/impresso_text_embedder/research/query_embed.py`;
design narrative, rejected alternatives (embed-inline-at-eval-
time, `embed_sweep --mode queries` flag, bundling into
`query_generate.py`, streaming read, pre-flight tokenise gate,
chunker-span persistence), and open items in
[`./query-embed/notes.md`](./query-embed/notes.md).

### 8 — eval-harness

Doc-level retrieval scoring + the per-study notebook that turns
the per-scenario doc embeddings (step 3) and the per-query
embedding shard (step 7) into the chunking-eval verdict. New
helper module
`src/impresso_text_embedder/research/eval.py` owns the testable
surface — S3 loaders with local-mirror caching
(`ensure_local`), pre-scoring sanity checks (`run_sanity` —
embedding-dim consistency, `ci_id` coverage, unit-norm spot-
check, lg breakdowns), the score pipeline (`score_queries` →
tidy DataFrame with `rank`/`reciprocal_rank`/`recall_at_k` per
`(query_id, scenario_id)`), and the Δ-vs-baseline table
(`baseline_delta_table` with paired-bootstrap CIs flagging
`beats_baseline`/`loses_to_baseline`). Per-language retrieval
pool only (fr queries vs fr docs); cross-lingual is the gated
O4/O18 ablation. Doc-level metrics only (binary Recall@k,
MRR); chunk-level IoU / Precision_Ω deferred to O15 because
chunker span-recovery doesn't exist on this branch. Tie-breaking
on `rank_of_gold` is pessimistic (competition-rank) so a
degenerate near-zero embedding scenario can't inflate Recall@1
by tying with a flat-zero pool. The notebook
(`notebooks/<study>-eval.ipynb`, generated from
`scripts/build_eval_notebook.py <study>` so a template tweak
lands in every per-study notebook with one rerun) is
the analysis surface — seaborn theme (colorblind palette,
white-grid context), per-language facets, S0 dashed reference
line on every primary plot, log₂ x-axis on the convergence
plot, symlog y-axis on the rank-of-gold boxplot. New deps under
the `[research]` extra: pandas, numpy, seaborn, matplotlib,
jupyterlab, pyarrow — kept off the default install so the
production Docker image stays lean. Headline plots: Recall@5 by
scenario (faceted by lg, hued by chunker), Recall@5 vs
`chunk_tokens` (convergence), position-bucket robustness, MRR,
query-type ablation (closes O9), rank-of-gold boxplot,
`n_chunks` distribution. Verdict lands in the notebook's last
markdown cell — three lines per language citing the Δ Recall@5
+ CI from the Δ-table. **No production CLI changes**: this
branch is research-only; findings ship as a recommendation,
not a default-flag flip. Code at
`src/impresso_text_embedder/research/eval.py` (20 unit tests
in `tests/test_research_eval.py`); generator at
`scripts/build_eval_notebook.py`; notebook for study-A-fit at
`notebooks/study-A-fit-eval.ipynb`. Design narrative, rejected
alternatives (CLI-only, all-inline notebook, papermill,
token-level Recall@k, NDCG, pre-computed CIs, single
multi-study notebook), and open items O15–O18 (chunk-level
metrics, verdict-cell automation, cross-study notebook,
cross-lingual ablation) in
[`./eval-harness/notes.md`](./eval-harness/notes.md).

### 9 — semantic-chunker-fixes

Three coupled defects in the semantic family (S11–S15) of the sweep,
all silent in pre-fix runs: (a) chonkie's `chunk_size` was measured
in `potion-base-8M`'s 30k-vocab WordPiece tokenizer rather than the
GTE multilingual SentencePiece tokenizer, so realized chunks were
~58% of the nominal target on French/German; (b) our
`SemanticStrategy` passed `min_sentences=` to chonkie 1.6.4, which
had renamed the kwarg to `min_sentences_per_chunk` — the floor was
silently `1` instead of the documented `5`; (c) `model2vec` was
missing from deps so chonkie warned and fell back to a slower
SentenceTransformer-based `potion-base-8M`. Fixes: rename the
kwarg in `chunking/semantic.py`, add an optional `tokenizer=` param
that swaps `_chunker._tokenizer` post-construction (chonkie's
public `tokenizer` is a read-only property and `SemanticChunker`
accepts no constructor override; `_tokenizer` is the documented
underlying attribute and the comment flags the private-API reach),
thread `tokenizer=model.tokenizer` from
`embed_sweep.build_long_doc_config` into the semantic-chunker
kwargs, and add `model2vec>=0.3` to the `[research]` extra. Three
new regression tests in `tests/test_chunking.py`. Out-of-sweep
behaviour (`SemanticStrategy()` with no `tokenizer=`) preserved
byte-for-byte. End-to-end smoke on a 464-GTE-token French passage
at `chunk_size=200`: realized GTE tok/chunk shifted from
`[116, 116, 116, 116]` to `[174, 174, 116]`. Remaining shortfall
(174 vs 200) is chonkie's normal soft-target slack, not a bug —
`n_tokens_per_chunk` already records realized sizes for post-hoc
binning. Rejected alternatives (per-language scaling at the call
site, custom `BaseEmbeddings` subclass, pinning to the ST
fallback) and follow-ups (re-run S11–S15 against the existing
study corpus to confirm the shift holds at scale; drop the
`_tokenizer` reach when chonkie ships a public setter) in
[`./semantic-chunker-fixes/notes.md`](./semantic-chunker-fixes/notes.md).

### 10 — aggregation-sweep

Realises the deferred "Length-weighted / max / first-chunk ablations" line
from [Scope](#scope-locked-with-the-user) early on `token-budget` rather
than waiting for a definitive winner from study-A-fit. New study
`C-aggregator` at `configs/research/study-C-aggregator.yaml` pins the chunker
to `token-budget` and sweeps all four registered aggregators (`mean`,
`max`, `first-chunk`, `length-weighted`) at three sizes
`[256, 1024, 4096]` — 13 scenarios total (1 truncate baseline + 1×3×4).
256 is intentionally a new chunk size not present in A-fit/B-overflow:
small chunks stress aggregation choice (more vectors to combine, more
weight asymmetry for length-weighted). Corpus filters identical to
A-fit so `corpus.jsonl.bz2` / `queries.jsonl.bz2` /
`queries-embedded.jsonl.bz2` server-side-copy via the new
`impresso-research-study-seed` CLI (`io.copy_s3_object` →
`s3.copy_object`, no body transfer, idempotent skip-if-exists).
Schema change: `ScenariosConfig.aggregators: tuple[str, ...] | None`
fans the cartesian into a third dimension when set, mutually exclusive
with the singleton `aggregator: str` form; singleton path stays
bit-identical so A-fit/B-overflow/v1 keep their scenario IDs and
`config_sha`. Scenario label grows the `-{agg}` suffix only when
`len(aggregators) > 1`. Output records gain an `aggregator` field for
downstream eval slicing (`(none)` for the truncate baseline). Code at
`src/impresso_text_embedder/research/{study_config,scenario_builder,study_seed}.py`
+ `src/impresso_text_embedder/io.py` (the `copy_s3_object` helper).
Design narrative, ordering rationale (size-major / agg-innermost so
"all aggregators at one cell" is a contiguous slice of IDs), and
rejected alternatives (separate-studies-per-aggregator,
fully-explicit `scenarios:` list, aggregator-major ordering) in
[`./aggregation-sweep/notes.md`](./aggregation-sweep/notes.md).

## Open items

- ~~O1 — corpus-sampling specification: decade window, OCR-quality cutoff,
  per-language target N~~ — closed by step 1; parameters frozen in
  [corpus-selection/notes.md](./corpus-selection/notes.md).
- ~~O2 — query-generation budget: GPT-4o calls, cost ceiling, cosine-filter
  thresholds~~ — closed by step 4: budget framed in tokens/cost
  (~$0.40 / ~20 min for 1200 generations on Qwen3-30B-A3B via RCP
  AIaaS), cosine filter deferred to O7. See
  [query-generation/notes.md](./query-generation/notes.md).
- ~~O3 — eval harness location~~ — resolved: research code lives under
  `src/impresso_text_embedder/research/` (chosen in step 1).
- O4 — cross-lingual ablation gating: only run if monolingual sweep produces
  a clear winner. Note: lb-as-target is corpus-blocked at uniform OCR≥0.9;
  lb-as-source for a fr-query→de-pool ablation remains feasible.
- O5 — chars_per_token calibration: refine the conservative 4.5/3.5
  estimates by tokenising the actual manifest text once it's fetched.
- O6 — de pre-1940 underrepresentation: 22 of 200 de docs are pre-1940
  under the uniform 0.9 OCR cutoff; decide whether to add a stratified-by-
  decade sampler or accept the skew.
- O7 — cosine filter for query quality (relevance + intra-doc dedup). Not
  applied at v1 since we have no retention-rate baseline yet to tune
  against. Promote to a follow-up step that hits the AIaaS embeddings
  endpoint (`Qwen/Qwen3-Embedding-8B`) if v1 inspection shows >10% of
  queries are off-topic or near-duplicates. See
  [query-generation/notes.md](./query-generation/notes.md) → O7.
- O8 — `stats.json` sidecar at `chunking-eval/queries/stats.json` with
  per-`(lg, bucket, query_type)` retention counts, real cost, and p50/p95
  latency for the run. Trivial follow-up; gated on v1 landing.
- O9–O11 — query-type ablation, multi-query-per-bucket, Qwen3 OCR-noise
  sanity-check. All described in the step-4 notes; non-blocking.
- O12 — eval harness inside step 6 consumes
  `queries-embedded.jsonl.bz2` (from step 7) + per-scenario
  `S{0..N}.jsonl.bz2` (from step 3), joins on `ci_id`, computes
  per-query doc-level Recall@k and chunk-vs-excerpt IoU /
  Precision_Ω from a deterministic chunker re-run keyed off
  `scenario_id`. Lives in step 6's (yet-to-open) notes folder.
- O13 — query-overflow telemetry: WARN when a query exceeds
  `model_max_tokens` on the encode path. Trivial; gated on a real
  step-7 run showing a non-zero count. Notes at
  [query-embed/notes.md](./query-embed/notes.md) → O13.
- O14 — re-embed queries under `Qwen/Qwen3-Embedding-8B` for the
  deferred O7 cosine filter. Reachable today via `--model-name` /
  `--model-revision` overrides on the step-7 CLI; promote to a
  follow-up step only when O7 is promoted out of "deferred".
  Notes at [query-embed/notes.md](./query-embed/notes.md) → O14.
- O15 — chunk-level IoU / Precision_Ω. Diagnostic for chunker
  boundary quality; needs a `chunk_with_spans()` helper that returns
  `(text, char_start, char_end)` per chunk under a given scenario.
  Doc-level metrics in step 8 fully answer the headline question;
  promote when a follow-up needs the deeper diagnostic. Notes at
  [eval-harness/notes.md](./eval-harness/notes.md) → O15.
- O16 — verdict-cell automation in the per-study notebook
  (auto-populate from the top Δ-table row). Notes at
  [eval-harness/notes.md](./eval-harness/notes.md) → O16.
- O17 — cross-study comparative notebook (`notebooks/cross-study-eval.ipynb`)
  loading both studies' `scores.parquet`. Gated on both runs landing.
  Notes at [eval-harness/notes.md](./eval-harness/notes.md) → O17.
- O18 — cross-lingual ablation (renamed from O4). Run the eval with
  `pool[pool.lg != query.lg]`. Gated on a monolingual winner. Notes at
  [eval-harness/notes.md](./eval-harness/notes.md) → O18.
- O19 — corpus-design changes for the next study-A-fit run: topic-bounded
  prefilter, n_per_lg 200→300, decade-stratified sampling. Raised by the
  2026-05-01 post-mortem on the first A-fit run; closes the ceiling-effect
  + temporal-skew problems surfaced there. Notes at
  [eval-harness/notes.md](./eval-harness/notes.md) → O19.
- O20 — alternative metrics in `eval.score_queries`: cosine score margin
  + softmax NLL alongside Recall@k. ~10 LoC reusing the cosine matrix
  already built; surfaces embedding-confidence signal Recall@k hides
  (e.g. semantic chunkers degrading NLL while looking competitive on
  Recall@5). Notes at [eval-harness/notes.md](./eval-harness/notes.md) → O20.
- O21 — hard-negative-restricted Recall@k as a free post-hoc equivalent
  of topic-bounding: pre-mine top-K=10 hardest non-gold docs per query
  via S0 baseline, freeze across scenarios, evaluate as
  (1+K)-classification. Lands alongside O20. Notes at
  [eval-harness/notes.md](./eval-harness/notes.md) → O21.
- O22 — promote O15 (token-level IoU / Precision_Ω à la Chroma) sooner.
  Reframed by the 2026-05-01 post-mortem as the *principled* chunker
  metric — pool-independent, so the topic-orthogonality problem
  disappears at the root rather than being mitigated by O20/O21. Notes
  at [eval-harness/notes.md](./eval-harness/notes.md) → O22.

## Adding a new step

1. **Pick the next number.** Step numbers are stable and never reused.
2. **Choose a slug.** Lowercase, hyphenated, descriptive (`gpu-profiles`,
   `validate-source-stats`). The slug is the notes-folder name.
3. **Decide if a notes folder is warranted.**
   - **Yes** when the step locks in a non-trivial decision, has rejected
     alternatives worth recording, or carries a step-specific open-items
     list. Most steps end up here.
   - **No** for trivial scaffolding (e.g. steps 1, 3, 8). The status row
     stays — the body just keeps its detail inline.
4. **If yes, create `.progress/<slug>/notes.md`** following the
   workflow in [Building the notes folder](#building-the-notes-folder)
   below. Mechanism + rationale belong there, **not** in plan.md.
5. **Add a row** to the [Status at a glance](#status-at-a-glance) table
   with status `wip` (or `todo` if not started yet).
6. **Add a step section** under [Steps](#steps) at the end (chronological
   order). Keep the body short: one-paragraph summary, link to the notes
   folder if any, optional **"Still in queue"** bullets only when this
   step has step-specific follow-ups not captured in the global
   [Open items](#open-items-needing-real-hardware--data) section.
7. **If the step locks in an architectural rule**, add a bullet under
   [`CLAUDE.md` → Decisions recorded](../CLAUDE.md#decisions-recorded).
   plan.md can then reference that decision by name without re-explaining
   it.
8. **Advance the status as the step lands**: `wip` → `done` (or `partial`
   if only a slice landed but the framework supports more, or `deferred`
   if the step is abandoned).

### Building the notes folder

A good notes folder is research-driven, not opinion-driven. The mechanism
and rationale recorded here are what future sessions (Claude or human)
read when they re-encounter the same fork — so the bar is "another agent
can re-derive the decision in 5 minutes", not "the author remembered why".
Workflow for a non-trivial step:

1. **Scope the question.** Two or three sentences: what is the user
   actually constraining, what's load-bearing vs. nice-to-have. Capture
   this *before* scanning anything — it's the lens for everything below.

2. **Scan the codebase in parallel.** Launch one or more `Explore`
   subagents in a single message (parallel tool calls) to map the
   relevant surface area: existing patterns, neighbouring registries,
   tests that would need to change, prior decisions in `CLAUDE.md`. Use
   tightly-scoped prompts so the agents return facts, not opinions.
   Trust their reports; don't repeat the same searches yourself.

3. **Look up 2026 best practice on the web.** Use `WebSearch` for current
   guidance from Anthropic, HuggingFace, and the relevant package
   vendors; check upstream issue trackers (GitHub issues, HF
   discussions) when the topic involves a third-party library. Cite the
   URLs in the notes file so future sessions can re-check freshness.
   Always put the year in the search query — knowledge cutoffs drift
   and "best practice" rotates.

4. **Read upstream source when the issue is in someone else's library.**
   `git show`, GitHub permalinks, or `pip download && tar xf`. Don't
   infer API behaviour from docstrings alone — past steps
   (`transformers-v5-regression`, `upload-integrity`,
   `drop-impresso-essentials`) only landed cleanly because we read the
   upstream code.

5. **Enumerate options.** Catalogue at least 2–3 alternatives even when
   one is obvious — the rejected ones are valuable to record. Use
   letters (A, B, C…) or numbers; keep each option a paragraph with
   trade-offs, not a sentence. The full strategy catalogue in
   [`long-doc-chunking/notes.md`](./long-doc-chunking/notes.md) (11
   chunkers + 8 aggregators) is the gold-standard pattern.

6. **Validate against existing constraints.** Cross-check the chosen
   option against [`CLAUDE.md`](../CLAUDE.md) → Non-goals, Decisions
   recorded, and Hardware target. If the new choice tightens or
   contradicts an existing rule, surface that explicitly — either
   update the rule (with justification) or explain why the new context
   overrides it.

7. **Record rejected alternatives.** One bullet each: option name +
   one-sentence reason it didn't win. Future sessions hitting the same
   fork shouldn't have to re-derive.

8. **Record open items.** What the step *does not* close — acceptance
   bars needing live hardware, calibration TODOs, follow-up steps that
   warrant their own future entry.

**File structure** (loose convention; match what neighbouring folders do):
TL;DR (3–5 lines) → scope/context → mechanism → rejected alternatives →
open items → upstream references with URLs.