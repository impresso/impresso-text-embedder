# embedding-sweep — design notes

Step 3 of `research/chunking-eval`: run every chunking strategy in scope
over the corpus shard built by step 2 (`corpus-fetch`), one Run:AI job
per scenario, write one `.jsonl.bz2` of doc-level embeddings per
scenario. This is the artefact the eval harness (a future step) will
score — it is not the eval itself.

## Inputs / outputs

- **Input**: the corpus shard produced by step 2 — by default
  `s3://140-processed-data-sandbox/chunking-eval/corpus/corpus-v1.jsonl.bz2`
  (overridable via `CORPUS_BUCKET` / `CORPUS_KEY` make variables or
  `--corpus-bucket` / `--corpus-key` CLI flags; `--local-corpus` skips
  the download for laptop iteration).
- **Output (per scenario)**:
  `s3://$(OUTPUT_BUCKET)/$(OUTPUT_PREFIX)/<scenario_id>_<label>/<corpus_basename>`
  e.g.
  `s3://140-processed-data-sandbox/chunking-eval/embeddings/S3_fixed-window-2048/corpus-v1.jsonl.bz2`.
- **Per-record schema**: production text-level fields
  `{ci_id, model_id, embedding, size, ts, ci_type}` + research
  metadata carried through from the manifest
  `{lg, year, provider, alias, ocrqa, len_chars}` + sweep-level
  annotations `{n_chunks, scenario_id, chunker, chunk_tokens}`. The
  research-only fields make stratified-by-language and
  chunked-vs-one-shot eval queries one-pass on the output shard.

## Scope (locked with the user)

- Only scenarios the **current** chunking + aggregation registries can
  build. No new chunkers, no `stride` kwarg on `FixedWindowStrategy`,
  no length-weighted/max/first-chunk aggregation ablations on this
  step.
- 16 scenarios. The grid is intentionally symmetric — every chunker
  family covers the same five chunk sizes — so cross-family
  comparisons at any size are first-class:

  | ID     | Chunker        | `chunk_tokens`                  |
  | ------ | -------------- | ------------------------------- |
  | S0     | *(truncate)*   | —                               |
  | S1–S5  | `fixed-window` | 512 / 1024 / 2048 / 4096 / 8190 |
  | S6–S10 | `token-budget` | 512 / 1024 / 2048 / 4096 / 8190 |
  | S11–S15| `semantic`     | 512 / 1024 / 2048 / 4096 / 8190 |

- Aggregator: `mean` for every chunked scenario (mean+L2; the post-mean
  L2 is applied inside `MeanPoolStrategy`, the per-chunk L2 by the
  encoder's `Normalize` module — see CLAUDE.md → "Two independent L2
  normalizations").
- One Run:AI job per scenario, one GPU per job, model replicated. No
  shared queue, no DDP. Idempotent at the S3 layer (each scenario
  writes to its own prefix).

## Mechanism

1. **Scenario registry** `research/scenarios.py` — single `Scenario`
   dataclass keyed by id. `chunk_tokens` doubles as both the
   `LongDocConfig.model_max_tokens` *trigger* and the chunker's
   nominal target size (see "Trigger threshold = chunk size" below).
   The Makefile reads `all_scenario_ids()` at evaluation time so
   adding/removing a scenario in code automatically reshapes
   `runai-submit-research-all`.
2. **Sweep CLI** `research/embed_sweep.py` (entry point
   `impresso-research-embed-sweep`) — the per-job glue:
   - load model once via `model.load_model` (production loader; same
     `f7d567e` revision pin, same xformers/unpad path),
   - build `LongDocConfig` for the scenario (`None` for S0 truncate),
   - download the corpus shard locally (or use `--local-corpus`),
   - read every record into memory (corpus is ≤ a few hundred docs by
     design, so the in-memory pass is cheap and gives us per-`ci_id`
     metadata lookup for the merge below),
   - pre-compute `n_chunks` per `ci_id` by replaying the chunker
     (duplicate work vs the encode path; see "Why duplicate chunker
     pass" below),
   - feed records to `embed.embed_records(level="text")` with the
     scenario's `EncoderConfig`,
   - merge research metadata onto each emitted dict and write
     to a local `.jsonl.bz2`,
   - upload to the per-scenario S3 prefix.
3. **Makefile orchestration** — `runai-submit-research SCENARIO=Sx`
   submits one job; `runai-submit-research-all` loops the registry.
   Each job invokes the sweep CLI with the scenario id; same image,
   same model pin, same per-job GPU/CPU/memory knobs as the production
   `runai-submit` flow used to expose.

### Trigger threshold = chunk size

The embedder's long-doc handling fires only when
`tokens(doc) > LongDocConfig.model_max_tokens`. Production wires
`model_max_tokens` to the model's actual context (8192) and a chunk
size of 8190, so chunking only happens for *very* long docs.

For the research question — "does forcing sub-8k chunking + mean+L2
pool produce better doc-level embeddings than one-shot encode at
8190?" — that wiring is a non-starter: at `chunk_tokens=512` it would
still skip every doc ≤ 8192 tokens, defeating the experiment.

`build_long_doc_config(scenario, model)` therefore sets *both*
`model_max_tokens` and the chunker's target to `scenario.chunk_tokens`.
Result: at `S1` (`fixed-window-512`), every doc longer than 512 tokens
is split into 512-token windows and aggregated; only docs ≤ 512
tokens go through one-shot. That is the comparison the eval needs.

S0 (truncate) keeps `LongDocConfig=None`; the production tokenizer
truncation path applies, identical to `--long-doc-strategy=truncate`
on the production CLI.

### Why duplicate chunker pass

`embed_records` does not surface `n_chunks` per ci_id — `_PendingText`
tracks `n_texts` internally but it is consumed during flush and not
re-exposed. For the eval we need the count anyway (chunked vs
one-shot is the load-bearing predictor for half the planned
analyses), so the sweep CLI runs the chunker a second time over the
same records before invoking `embed_records`.

The cost is one extra tokenisation + chunk pass. For
`fixed-window` / `token-budget` the duplicate pass is milliseconds; for
`semantic` it pays one extra forward through chonkie's
`potion-base-8M` sentence embedder per long doc — a few seconds per
scenario over a 400-doc corpus. Acceptable for a research script that
runs at most 16 times total.

## Rejected alternatives

- **Add `n_chunks` to the production text-level output schema** — would
  force a write to every production embedding shard for a metric only
  the research path cares about. Rejected: production schema is
  governed by Impresso's `embeddings-docs.schema.json` (CLAUDE.md →
  "Decisions inherited from the migration"); changing it is a bigger
  decision than this branch is allowed to make.
- **Side-channel callback into `TextBatcher`** to record chunk counts
  per ci_id mid-flush — workable but pollutes the production hot path
  for a research-only feature. The duplicate pass keeps the production
  embedder untouched.
- **Embed the whole corpus once and re-aggregate per scenario** — the
  per-chunk vectors are not the same across scenarios because each
  chunker produces different chunks (different boundaries, different
  text). Aggregation alone cannot recover the "what is each scenario's
  doc-level embedding?" output. This was a brief temptation; ruled
  out by remembering that the chunker affects the encoder's *input*,
  not just its post-processing.
- **One scenario per job × per language** (32 jobs instead of 16) —
  language is a record-level field, not a job-level shard. Splitting
  in the embedding pass would only save eval-side filtering by `lg`,
  which is one column on the output. Rejected: doubles the GPU
  footprint for no measured benefit.
- **Submit all scenarios as one runai job** that loops internally —
  appealing for shared model load (16x avoided), but a single job
  failing late takes the whole sweep with it; one-job-per-scenario
  gives independent retry semantics matching the file-level sharding
  pattern the production embedder already uses
  (`.history/multi-gpu-sharding/notes.md`). Model load is ~30s vs
  hours of encode; the sharing benefit is small.
- **Extend `impresso-embed-create` with a `--single-input` flag** to
  reuse the production CLI directly — would entangle research-only
  configuration (per-scenario `model_max_tokens` override, output
  schema augmentation) with the production CLI's flag surface. The
  branch goal explicitly forbids production-pipeline changes until
  findings justify a default change (CLAUDE.md → "Out of scope on
  this branch"). A separate `impresso-research-embed-sweep` keeps the
  research path contained to `research/`.

## Open items

- **`n_chunks` distribution per scenario** is logged as bucketed counts
  in the per-run summary line (`=1 / 2-4 / 5-16 / 17+`); a deeper
  analysis of where each chunker lands lives in the eval step (TBD).
- **Cross-scenario diff** — quick sanity check that S0's embedding for
  a doc ≤ 8192 tokens equals S6's embedding for the same doc (both
  paths should produce a one-shot encode for short docs). Not gated
  on landing this step but would catch wiring regressions early.
- **Wallclock per scenario on RCP** — the dispatch uses
  `--gpu 1`; the 16 jobs may queue or run concurrently depending on
  cluster availability. Calibrate at first real submission and revisit
  the `runai-submit-research-all` body if serial submission becomes a
  bottleneck (today it just loops `make runai-submit-research`).
