​# query-embed — design notes

Step 7 of `research/chunking-eval`: materialise per-query embeddings
once into a sidecar `queries-embedded.jsonl.bz2` so the eval step
inside step 6 can consume them without re-loading the embedding
model. Mirrors `research/embed_sweep.py`'s infra — same study YAML,
same Run:AI scaffolding, same `staged_input`/`staged_output`
helpers, same pinned model — but reads the **queries** shard from
step 4 instead of the corpus, and skips chunking + aggregation
entirely (queries are short plain strings, never long-doc).

## Inputs / outputs

- **Input**: the queries shard built by step 4,
  `s3://{study.s3.bucket}/{study.s3_root}/queries.jsonl.bz2`
  (one record per query). Each record carries
  `{query_id, ci_id, lg, query_text, query_type, position_bucket,
  position_chars, references, gen_model, gen_endpoint, ts,
  study_name?, study_config_sha?}`.
- **Output**: a single `.jsonl.bz2` of embedded queries at
  `s3://{study.s3.bucket}/{study.s3_root}/queries-embedded.jsonl.bz2`
  (one record per query — same row count as the input).
- **Per-record schema**:

  ```
  {
    query_id,                 // primary key — stable across runs
    ci_id,                    // foreign key to corpus doc
    embedding,                // list[float], 5-dp rounded (TextRecord shape)
    size,                     // int, vector dim (768 for gte-mb)
    model_id,                 // build_embedder_tag(name, revision)
    ts,                       // UTC YYYY-MM-DDTHH:MM:SSZ

    // Query metadata mirrored verbatim — kept on the same row so
    // the eval can stratify in one pass without joining back to
    // queries.jsonl.bz2:
    lg, query_type, position_bucket, position_chars,
    references,               // [{text, char_start, char_end}, ...]
    query_text,               // raw string, kept for inspection

    // Provenance (mirrors embed_sweep):
    study_name, study_config_sha
  }
  ```

  The eval step in step 6 reads this file once, builds a
  `query_id → embedding` map plus a `query_id → references` map,
  and joins against per-scenario doc embeddings on `ci_id`.

## Scope (locked with the user)

- **One CLI per study, one S3 file per study.** Output filename
  is fixed at `queries-embedded.jsonl.bz2`; the queries are
  embedded under the **same model + revision** declared in the
  study YAML's `embed:` block, the same one `embed_sweep` consumes.
  Re-embedding under a *different* model (e.g. for the deferred
  O7 cosine filter with `Qwen/Qwen3-Embedding-8B`) is reachable via
  `--model-name` / `--model-revision` overrides exactly as
  `embed_sweep` exposes them — but the canonical artefact under
  the study prefix is the run that matches the study's pinned
  embedder.
- **No chunking, no aggregation, no record filtering.** Queries
  are short LLM-generated strings (typical 20–50 tokens), never
  exceed `model_max_tokens`. Skipping `embed_records` (which
  carries content-type / sentence-presence / long-doc routing
  semantics that don't apply) keeps the path explicit. Direct
  call to `model.encode_texts(list[str], batch_size, precision)`.
- **Same Run:AI / log scaffolding as `embed_sweep`.** New Make
  target `runai-submit-query-embed STUDY=<name>`; one job per
  study; logs land at
  `/rcp-scratch/<user>/experiments/chunking-eval/<study>/<YYYY-MM-DD>/query-embed.log`
  with the same `--log-dir` override and PVC fail-fast as
  `embed_sweep`.
- **Single embed pass, in-memory.** The queries shard is at most
  ~2400 records (400 docs × 3 buckets × 2 query types × N samples
  per bucket-type cell, default 1). Read into memory, encode in
  one batched call, write back. No streaming.
- **Per-record provenance.** `study_name` + `study_config_sha`
  are written on every output record so two studies cannot share
  rows undetected, and a config edit re-fingerprints. Same
  invariant `embed_sweep` already enforces.

## Mechanism

1. **Parse `--config <study.yaml>`** → `StudyConfig` via
   `load_study_config`. Resolve embedder knobs (`model_name`,
   `model_revision`, `precision`, `attention`, `unpad_inputs`,
   `batch_size`) using the same `_resolve_value(cli, cfg, fallback)`
   pattern as `embed_sweep`.
2. **Configure logging** under
   `experiments/chunking-eval/<study>/<YYYY-MM-DD>/query-embed.log`
   via `configure_logging(provider="query-embed", …)`. Reuses
   `_resolve_research_log_dir` from `embed_sweep` (lift to a
   shared helper if a third caller appears; one-off duplication
   is fine for now per CLAUDE.md "three similar lines is better
   than a premature abstraction").
3. **Stage input** via `staged_input(bucket, queries_key)` where
   `queries_key = study_cfg.s3_key(QUERIES_FILENAME)`. Fail fast
   if the file is absent (step 4 must have run first).
4. **Read into memory.** `bz2.open(path, "rb")` + `orjson.loads`
   per line; collect to `list[Query]` (a thin dataclass mirror of
   the input fields, or a `dict` if a dataclass adds no value).
   Apply `--limit N` for smoke runs.
5. **Load the model** via `load_model(name, revision, …)`,
   identical to `embed_sweep`. The load-time `Normalize`-module
   assertion ensures unit-norm output without a redundant L2
   here.
6. **Encode** in a single call:
   `vectors = encode_texts(model, [q.query_text for q in queries],
   batch_size=cfg.batch_size, precision=cfg.precision,
   show_progress_bar=False)` (we drive a `tqdm` on the *write*
   loop instead, matching `embed_sweep`).
7. **Build output records**: per-query dict combining the
   query-side metadata (verbatim) + `embedding` (5-dp rounded
   list, mirroring `schema.to_dict`'s text-level rounding) +
   `size` + `model_id` (= `build_embedder_tag`) + `ts` (=
   `utc_timestamp`) + `study_name` + `study_config_sha`.
8. **Write & upload** via `staged_output(bucket, out_key,
   local_mirror, upload=not args.no_upload)`. Output filename
   `QUERIES_EMBEDDED_FILENAME = "queries-embedded.jsonl.bz2"`,
   added to `study_config.py` next to the existing
   `QUERIES_FILENAME` constant.
9. **Stats line**: `query-embed stats: input=N embedded=N
   skipped_empty=K dim=768 model=<tag>`. Same INFO-line shape as
   `embed_sweep`'s `_format_stats`.

## Rejected alternatives

- **Embed queries inline at eval time** (eval CLI loads the
  model, encodes queries, computes scores in one shot). Rejected:
  couples the eval to a GPU/RCP scheduling round-trip; re-runs
  the encode every time eval iterates (e.g. when sweeping query
  cosine thresholds for O7); mixes "compute" and "score" in one
  CLI. Materialising once and reading a `.jsonl.bz2` is the
  separation that lets the eval step run on CPU off-RCP if
  needed.
- **Reuse `embed_sweep` with a `--mode {corpus,queries}` flag.**
  Rejected: `embed_sweep`'s contract is "embed the corpus under
  one *scenario*"; queries have no scenario, no chunking, no
  long-doc routing. The flag would force half of `embed_sweep`'s
  body into branches and pollute its YAML schema. Cleaner to
  ship a sibling CLI that shares only the load-model + staging
  primitives. Same call already explicitly made in
  `embedding-sweep/notes.md` for "do not bend the existing CLI".
- **Bundle query embedding inside `query_generate.py`** (LLM
  call + embed in one job). Rejected: `query_generate` runs on
  CPU and only depends on `langchain-openai`; pulling
  `sentence_transformers` + the pinned model into that job
  forces a GPU step that today is genuinely separable, and
  blocks the future case where we re-embed under a different
  model without re-spending the LLM budget.
- **Stream-read queries instead of in-memory.** Rejected: the
  shard is ≤2400 records (~5 MB bz2 worst case); streaming buys
  nothing and complicates the "encode once, write once"
  pattern. `embed_sweep` reads the corpus in-memory via
  `_read_corpus` for the same reason at the same scale.
- **Skip persisting `query_text` on the embedded shard** (it's
  already on `queries.jsonl.bz2`). Rejected for the same reason
  the embedded record duplicates `lg` / `position_bucket` /
  `references`: the eval consumes one file, stratifies in one
  pass, and never has to do an O(N) join back to the queries
  shard. Storage cost is negligible (~100 chars per query × 2400
  rows ≈ 240 KB before bz2).
- **Tokenise queries pre-flight** to assert none exceed
  `model_max_tokens`. Rejected as a hard gate: queries are
  bounded by the LLM-emitted length cap (~1500 output tokens
  worst-case at step 4) and fall well under 8192 by construction.
  A WARN-on-overflow telemetry log is fine and could land as a
  follow-up; a hard gate would be theatre.
- **Persist the chunker's `Chunk.start` spans alongside the
  embedding** (would let chunk-vs-excerpt IoU read straight from
  the embed-sweep output). Rejected for *step 7* — that span
  belongs on the *corpus* side, not the queries side. The eval
  re-runs the chunker deterministically from `scenario_id` at
  eval time (O(ms) per doc on 400-doc corpus); the queries
  shard's job is just to give the eval a pre-encoded query
  vector per `query_id`. Listed under [step 6's
  pre-conditions](../plan.md#6--study-a-fit--study-b-overflow-runs)
  as the eval-side mechanism.

## Open items

- **O12 — eval harness consumes this artefact.** The eval step
  in step 6 reads `queries-embedded.jsonl.bz2` + per-scenario
  `S{0..N}.jsonl.bz2`, joins on `ci_id`, computes per-query
  doc-level Recall@k against the scenario's doc pool, and
  computes chunk-vs-excerpt IoU / Precision_Ω from the chunker
  re-run. Lives in step 6's notes folder when that step opens.
- **O13 — query-overflow telemetry.** WARN when a query exceeds
  `model_max_tokens` (silently truncated by the tokenizer
  otherwise). Trivial; gated on a real run showing a non-zero
  count.
- **O14 — re-embed under a second model for O7 cosine filter.**
  The deferred query-quality filter (CLAUDE.md → step 4 →
  Open items → O7) wants `Qwen/Qwen3-Embedding-8B`. Either pass
  `--model-name`/`--model-revision` overrides on this CLI and
  write to `queries-embedded-qwen3.jsonl.bz2`, or grow a
  `--output-suffix` knob. Punt until O7 is promoted out of the
  deferred bucket.

## Reproducing

```
# Embed the queries for the current study, against the study's
# pinned embedder:
uv run impresso-research-query-embed \
  --config configs/research/study-A-fit.yaml

# Smoke run on the local corpus mirror (no S3 round-trip):
uv run impresso-research-query-embed \
  --config configs/research/study-A-fit.yaml \
  --no-upload \
  --limit 16

# Run on RCP under Run:AI:
make STUDY=study-A-fit runai-submit-query-embed
```

## Upstream references

- `research/embed_sweep.py` — sibling CLI; this step mirrors its
  scaffolding choices (study YAML resolution, log dir,
  `staged_input`/`staged_output`, `_resolve_value` precedence).
- `research/query_generate.py` — produces the input shard;
  `Query` dataclass at the top of that file is the source of
  truth for the per-query field set we mirror onto the embedded
  shard.
- CLAUDE.md → "Decisions inherited from the migration" →
  "`--normalize-embeddings` removed; encoder must ship
  `Normalize` module" — load-time invariant that lets us call
  `encode_texts` without an explicit L2 step.
