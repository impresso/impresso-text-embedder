# Migration plan — impresso-text-embedder

## What this file is

The chronological ledger of the migration from the Make/script layout on
`main` to the Python package on `feat/migration-python-package`. Step
numbers are stable; statuses advance as each step lands.

- **Not the rulebook.** Current architectural rules live in
  [`CLAUDE.md`](../CLAUDE.md), under
  [Decisions recorded](../CLAUDE.md#decisions-recorded).
- **Not the design narrative.** Each numbered step links to a
  `.progress/<slug>/` notes folder where the mechanism, rationale,
  rejected alternatives, and step-specific open items live.
- **How to read it.** Scroll the status table for the at-a-glance picture,
  then click into a step. Each step body is a one-paragraph summary with
  pointers — depth lives in the linked folder.

Statuses: `done` · `partial` · `wip` · `todo` · `deferred`.

## Status at a glance

| #  | Step                                                      | Status     | Notes folder                                                  |
| -- | --------------------------------------------------------- | ---------- | ------------------------------------------------------------- |
| 1  | [package-skeleton](#1-package-skeleton)                   | `done`     | —                                                             |
| 2  | [io-layer](#2-io-layer)                                   | `done`     | [`io-layer/`](./io-layer/)                                    |
| 3  | [schema-text-rebuild](#3-schema-text-rebuild)             | `done`     | —                                                             |
| 4  | [model-encoder](#4-model-encoder)                         | `done`     | [`gpu-throughput/`](./gpu-throughput/)                        |
| 5  | [chunking](#5-chunking)                                   | `done`     | [`chunking/`](./chunking/)                                    |
| 6  | [create-cli](#6-create-cli)                               | `done`     | [`create-cli/`](./create-cli/)                                |
| 7  | [validate-cli](#7-validate-cli)                           | `done`     | [`validation-metric/`](./validation-metric/)                  |
| 8  | [e2e-docs](#8-e2e-docs)                                   | `done`     | —                                                             |
| 9  | [docker-runai](#9-docker-runai)                           | `done`     | [`docker-runai/`](./docker-runai/)                            |
| 10 | [reembed-on-change](#10-reembed-on-change)                | `done`     | [`reembed-on-change/`](./reembed-on-change/)                  |
| 11 | [gpu-profiles](#11-gpu-profiles)                          | `done`     | [`gpu-profiles/`](./gpu-profiles/)                            |
| 12 | [drop-impresso-essentials](#12-drop-impresso-essentials)  | `done`     | [`io-layer/`](./io-layer/)                                    |
| 13 | [io-throughput](#13-io-throughput)                        | `done`     | [`io-throughput/`](./io-throughput/)                          |
| 14 | [model-revision-pin](#14-model-revision-pin)              | `done`     | [`model-revision-pin/`](./model-revision-pin/)                |
| 15 | [structured-logging](#15-structured-logging)              | `done`     | [`structured-logging/`](./structured-logging/)                |
| 16 | [long-doc-chunking](#16-long-doc-chunking)                | `partial`  | [`long-doc-chunking/`](./long-doc-chunking/)                  |
| 17 | [validate-source-stats](#17-validate-source-stats)        | `done`     | [`validate-source-stats/`](./validate-source-stats/)          |
| 18 | [multi-gpu-sharding](#18-multi-gpu-sharding)              | `done`     | [`multi-gpu-sharding/`](./multi-gpu-sharding/)                |

## Currently active

- [Step 16 — `long-doc-chunking`](#16-long-doc-chunking) (`partial`):
  the framework + `fixed-window` chunker + `mean` aggregation shipped.
  Additional strategies are small follow-ups — each is one module + one
  `register_strategy` call + one `choices=` entry.

## Open items needing real hardware / data

Per-step acceptance items that require live measurement, not code:

- **Step 11**: per-profile batch-size calibration; confirm FA3 fires on
  H100; A100↔H100 `--tol 1e-4` cross-check; record RCP node-type labels.
- **Step 13**: confirm GPU SM utilization ≥85% during steady-state encode.
- **Step 16**: real-data calibration of `--long-doc-chunk-tokens`;
  per-language `chars_per_token` for the fast gate; long-doc query-set
  recall vs. the truncate baseline.
- **Step 17**: per-`lg` / per-`tp` mean-drift breakdowns inside the
  VALUE panel; drift-vs-length correlation; `--source-samples N` flag.
- **Step 18**: 4-shard real-RCP run on the largest provider; shard
  wallclock skew <1.5× to keep round-robin, otherwise promote
  size-aware greedy.

## Steps

### 1. package-skeleton

`done` · no notes folder

Initial scaffolding: `pyproject.toml` (hatchling + uv), `src/impresso_text_embedder/`, Python ≥ 3.10, ruff config, `LICENSE` (AGPL-3.0-or-later, mirrored from `main`), `tests/` placeholder, gitignore update, `.flake8` removed. Verified by `uv sync` resolving cleanly and the placeholder test passing.

### 2. io-layer

`done` · [`io-layer/`](./io-layer/)

Streaming S3 reader for `.jsonl.bz2` shards, provider/alias/year enumeration, idempotent skip-if-exists. `impresso_essentials.io.s3.read_jsonlines` deliberately bypassed because it slurps the whole file into memory and breaks the prefetch model. Dotenv loaded at the CLI boundary only.

### 3. schema-text-rebuild

`done` · no notes folder

Ports `rebuild_ft_from_offsets` and `rebuild_sentence_from_offsets` verbatim from `main:lib/text_embedding_processor.py`. Typed schemas for the three output shapes (text / sentence / chunk).

### 4. model-encoder

`done` · [`gpu-throughput/`](./gpu-throughput/)

`SentenceTransformer` load + bf16 autocast around `encode`, fp32 weights, `inference_mode`, `trust_remote_code=True` for `gte-multilingual-base`. See CLAUDE.md decision **"A100 bf16 strategy"**. Real-GPU throughput measurement landed later (steps 11 and 13).

### 5. chunking

`done` · [`chunking/`](./chunking/)

Kwargs-capable `text → K-chunks` registry. Ships `semantic` (chonkie `SemanticChunker` at threshold `0.5`, chunk size `1024`, min sentences `5`) for `--embedding-level=chunk`. Step 16 later extends the same registry with `fixed-window` and `token-budget` for the long-doc text path.

### 6. create-cli

`done` · [`create-cli/`](./create-cli/)

`impresso-embed-create --provider …` CLI entry point + orchestrator: 1:1 input→output mapping, model load, encode, upload. Output schema aligned to the Impresso document-embeddings spec — required `{ci_id, model_id, embedding, size}`, optional `{ts, ci_type}`. See CLAUDE.md decision **"Text-level output schema aligned with Impresso document-embeddings schema"**. The async prefetch + upload overlap originally planned for this step actually landed in step 13.

### 7. validate-cli

`done` · [`validation-metric/`](./validation-metric/)

`impresso-embed-validate <path> [--target] [--tol]`. Metric: cosine distance on L2-normalized vectors; default tolerance `1e-4`. See CLAUDE.md decision **"Validation metric"**. Step 17 later extends this CLI with `--source` diagnostics + Rich rendering.

### 8. e2e-docs

`done` · no notes folder

Tiny local fixture run end-to-end through `impresso-embed-create`; first README polish; pruned the "Things to decide" list in CLAUDE.md, splitting it into "Decisions recorded" and "Still open — needs real GPU time".

### 9. docker-runai

`done` · [`docker-runai/`](./docker-runai/)

Container image (`nvcr.io/nvidia/pytorch:25.03-py3`) with LDAP-matched user for PVC ownership; `ENTRYPOINT ["impresso-embed-create"]`. Slim `Makefile` for docker build/push, k8s secret creation, runai submit + interactive debug — no data-processing logic in the Makefile.

### 10. reembed-on-change

`done` · [`reembed-on-change/`](./reembed-on-change/)

Skip-decision compares S3 `LastModified` of input vs. existing output; re-embeds when input is newer. `--force` overrides unconditionally. See CLAUDE.md decision **"Re-embed on input change"**. Known gap: byte-identical re-uploads still trigger a re-embed; this does not replace the deferred `impresso_essentials.versioning` manifest system.

### 11. gpu-profiles

`done` · [`gpu-profiles/`](./gpu-profiles/)

Same image runs on A100 and H100/H200. `accel.py` detects capability at model-load time and picks the per-profile default batch size; xformers' `memory_efficient_attention` dispatches FA2/FA3 transparently from inside the bf16 autocast region. `unpad_inputs` + `use_memory_efficient_attention` reach the model config via `config_kwargs` (not `model_kwargs` — ST v5 pre-loads the config). See CLAUDE.md decisions **"GPU profile detection"** and **"xformers + unpadding wired as the default fast path"**.

### 12. drop-impresso-essentials

`done` · [`io-layer/`](./io-layer/) (shared with step 2)

Removed the `impresso-essentials` runtime dependency entirely; vendored the three S3 helpers we used (`get_s3_client`, `get_s3_resource`, `upload_to_s3`) as ~40 lines of boto3 wrappers in `src/impresso_text_embedder/io.py`. See CLAUDE.md decision **"impresso-essentials vendored, not imported"**.

### 13. io-throughput

`done` · [`io-throughput/`](./io-throughput/)

Closes the CPU/IO half of the "GPU must be the bottleneck" target: prefetch + upload overlap via two single-slot `ThreadPoolExecutor`s, `json` → `orjson` on read and write, multipart S3 transfers (`TransferConfig`: 8 MB / 8 MB / 10 threads), and per-file telemetry (`download_s` / `encode_s` / `upload_wait_s` / `gpu_util_mean,p10`). See CLAUDE.md decisions **"Per-provider pipeline overlap"**, **"JSON codec"**, **"Per-file telemetry"**.

### 14. model-revision-pin

`done` · [`model-revision-pin/`](./model-revision-pin/)

`Alibaba-NLP/gte-multilingual-base` pinned to revision `f7d567e`. Single source of truth: `DEFAULT_MODEL_REVISION` in `src/impresso_text_embedder/model.py`. The `Makefile` mirrors the pin and `runai-submit` forwards it as an explicit `--model-revision` so the pin is recoverable from `runai describe job`. The output slug stays revision-agnostic by design. See CLAUDE.md decision **"Model revision pinned to `f7d567e`"**.

### 15. structured-logging

`done` · [`structured-logging/`](./structured-logging/)

`impresso-embed-create` splits its output: full INFO log to `/rcp-scratch/<user>/experiments/embeddings/<YYYY-MM-DD>/<provider>.log` (override with `--log-dir`); terminal gets a `tqdm` bar with per-file postfix (`dl=…s enc=…s up=…s gpu=…%`) and ERROR records only. Fail-fast when neither `/rcp-scratch` nor `--log-dir` is available. See CLAUDE.md decision **"Structured logging split file ↔ terminal"**.

### 16. long-doc-chunking

`partial` · [`long-doc-chunking/`](./long-doc-chunking/)

Two orthogonal kwargs-capable registries — `chunking` (text → K chunks) and `aggregation` (K vectors → 1 vector). Default long-doc path at `--embedding-level=text`: `fixed-window` chunker + `mean` aggregation; long docs are no longer silently truncated. `--long-doc-strategy truncate` restores pre-step-16 behaviour. Boundary: chunking only fires when `tokens(doc) > model_max_tokens` (one-shot wins for ≤8192-token docs because the encoder is CLS-pooled). See CLAUDE.md decision **"Long-doc handling at `--embedding-level text`"**.

**Still in queue** (each lands as one module + one `register_strategy` line + one `choices=` entry):

- Aggregation strategies: length-weighted mean, max pool, first-chunk, position-weighted, attention-weighted.
- Chunking strategies: stride overlap, paragraph packer, recursive, chonkie `Token` / `Sentence`.
- Token-budget chunker: prefer `record["sents"]` when present (extends `ChunkingStrategy.chunk` or builds the chunker per-record).
- Telemetry: surface `n_chunks` on `TextRecord` (gated on schema `additionalProperties` policy).
- Calibration: `--long-doc-chunk-tokens` sweep on real long docs; per-language `chars_per_token` for the fast gate.
- Goldens: regenerate the long-doc subset of any integration goldens to match the new default.

### 17. validate-source-stats

`done` · [`validate-source-stats/`](./validate-source-stats/)

`impresso-embed-validate --source <path>` cross-references the input shard with each of the three mismatch buckets — above-tolerance, missing-in-target, missing-in-produced — and emits per-direction stats (char-length log-scale histogram, `lg` / `tp` breakdowns, sample excerpts, worst-drift cosine distances) via Rich panels. ANSI auto-strips on non-TTY so legacy substring contracts in tests keep passing. `ValidationReport.passed` and exit codes are unchanged. See CLAUDE.md decision **"Validate — source-backed diagnostics (drifted + missing) + Rich rendering"**.

### 18. multi-gpu-sharding

`done` · [`multi-gpu-sharding/`](./multi-gpu-sharding/)

Horizontal throughput via file-level data parallelism: `--shard-index i --num-shards N` on `impresso-embed-create`, round-robin selection (`enumerate(keys) % N == i`) over `list_objects_v2`'s lexicographic output, applied lazily at `pipeline._plan_files`. One runai job per shard, one GPU per job, model replicated across jobs — no DDP/FSDP/tensor-parallel, no NCCL, no coordination beyond the static `(i, N)` partition. Re-runs are idempotent via the existing `--skip-if-s3-exists` + `reembed-on-change` `LastModified` check. Lifts the CLAUDE.md → Non-goals "Multi-GPU/DDP out of scope" bar **for the data-parallel case only**; multi-node and model-parallel mechanisms remain out of scope. Makefile gains `runai-submit-shard` (single shard) + `runai-submit-multi NUM_SHARDS=N` (loop). See CLAUDE.md decision **"Multi-GPU throughput via file-level sharding"**.

**Still in queue** (gated on the first real 4-shard RCP run):

- Calibration: shard wallclock skew on the largest provider; promote size-aware greedy if skew >2×.
- Per-shard manifest INFO line at startup (file count + first/last keys for `runai describe job` audit).
- Per-shard log filename (`<provider>-shard-i-of-N.log`) so concurrent shards don't stomp each other.
- Cross-shard telemetry aggregator (operator-side, not code).

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

## Related notes folders (not tied to a numbered step)

Post-migration decisions that don't have their own step. Linked here so
readers starting from plan.md can discover them:

- [`normalize-flag-removal/`](./normalize-flag-removal/) — `--normalize-embeddings` flag removed (CLAUDE.md decision: **"--normalize-embeddings removed; encoder must ship Normalize module"**).
- [`record-filtering/`](./record-filtering/) — record-filtering reason taxonomy and `missing_content_type` carve-out (CLAUDE.md decision: **"Record filtering — `missing_content_type` distinct from `content_type`"**).
- [`transformers-v5-regression/`](./transformers-v5-regression/) — why `transformers` is pinned `<5` (CLAUDE.md decision: **"Transformers pinned `<5`"**).
- [`upload-integrity/`](./upload-integrity/) — Ceph `MissingContentLength` workaround + post-upload verification (CLAUDE.md decision: **"Upload integrity"**).
