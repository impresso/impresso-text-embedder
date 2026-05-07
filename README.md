# impresso-text-embedder — `research/chunking-eval`

> **Side-research branch.** Chunking-strategy evaluation for
> [Impresso](https://impresso-project.ch) document embeddings. The
> production embedder ships from `main`; this branch never embeds the
> full Impresso corpus and is **not intended to merge back**.

Locked scope, research question, and inherited decisions in
[`CLAUDE.md`](./CLAUDE.md). Live ledger of work on this branch in
[`.progress/plan.md`](./.progress/plan.md).

## Workflow

```
corpus_select  →  corpus_fetch  ──────────────────────────→  embed_sweep
  manifest        corpus.jsonl.bz2                            N jsonl shards
                       │                                      (1 per scenario)
                       ↓
                  query_generate  →  query_embed
                  queries.jsonl.bz2  queries-embedded.jsonl.bz2
                  (synthetic LLM)    (1 vector per query)
```

Every stage reads a **single study YAML** under `configs/research/`.
The shipped studies:

| Study                | Corpus filter            | Scenarios | Question |
| -------------------- | ------------------------ | --------- | -------- |
| `study-A-fit`        | `7000 ≤ tokens ≤ 8000`   | 13        | Sub-context chunking vs one-shot — no-loss baseline. |
| `study-B-overflow`   | `tokens ≥ 16384`         | 16        | Of the strategies forced to do something, which loses least? |
| `study-C-aggregator` | `7000 ≤ tokens ≤ 8000` (reuses A-fit corpus) | 13 | Holding chunker fixed at `token-budget`, which aggregator (mean / max / first-chunk / length-weighted) recovers most signal? |

Schema in `src/impresso_text_embedder/research/study_config.py`.

## Install

Requires **Python ≥ 3.10**. Pinned with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
cp .env.example .env   # then fill in SE_* and RCP_API_KEY
```

`pip install -e '.[dev]'` works too. `pytest` + `ruff check .` for the
test suite.

## Run a study

Pick a study with `STUDY=...`. All four targets forward
`--config configs/research/$(STUDY).yaml`.

```bash
# 1 — laptop / login node
make STUDY=study-A-fit research-corpus-select
make STUDY=study-A-fit research-corpus-fetch
make STUDY=study-A-fit research-query-generate          # AIaaS (default)
# or, on RCP / Run:AI when you want a self-contained run:
make STUDY=study-A-fit runai-submit-query-generate-local

# 2 — list the registry derived from the study YAML
make STUDY=study-A-fit research-list-scenarios

# 3 — RCP / Run:AI: one runai job per scenario
make STUDY=study-A-fit runai-submit-research-all
# or one specific scenario:
make STUDY=study-A-fit runai-submit-research SCENARIO=S3

# 4 — RCP / Run:AI: encode the queries once into a per-query
# sidecar that the eval step joins against. One job per study.
make STUDY=study-A-fit runai-submit-query-embed
```

### Query generation: AIaaS or CaaS

Step 4 ships **two interchangeable backends** for the synthetic
`(query, gold_excerpts)` set. Both read the same study YAML, share
the same prompts, the same `QueryOutput` schema, the same
verbatim-anchor verification, and write the same
`queries.jsonl.bz2` shape — the eval step downstream cannot tell
them apart beyond the `gen_endpoint` field on each record.

| Backend                             | Where the LLM runs                                                                | Auth needed                            | Wallclock (study-A-fit, ~3600 generations) | When to pick                                                                                          |
| ----------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **AIaaS** (default)                 | EPFL RCP AIaaS endpoint (`https://inference.rcp.epfl.ch/v1`, OpenAI-compatible)   | `RCP_API_KEY` in `.env`                | ~60–100 min at the 2-parallel cap       | Laptop / login-node runs; you don't need a GPU; AIaaS is up and not throttled.                        |
| **CaaS** (`*-local` Make targets)   | Local `transformers.AutoModelForCausalLM` (bf16 + SDPA Flash-Attn-2) inside the production Run:AI image | none (model weights cached in `HF_HOME`) | ~1–2 h on a single H100 80GB at `batch_size=4` (study-A-fit) | RCP / Run:AI run that needs to be self-contained; AIaaS throttled or unavailable; reproducibility-locked study runs. |

```bash
# AIaaS — laptop / login node
make STUDY=study-A-fit research-query-generate

# CaaS — Run:AI (one H100 80GB job per study, batch_size=8 by default)
make STUDY=study-A-fit runai-submit-query-generate-local

# CaaS — laptop / GPU host smoke test (single-stream, batch_size=1)
make STUDY=study-A-fit research-query-generate-local \
     QUERY_GEN_LOCAL_ARGS="--no-upload --limit 1"
```

Both targets honour `STUDY=...` for the study YAML, and forward
extras via `QUERY_GEN_ARGS=` (AIaaS) and `QUERY_GEN_LOCAL_ARGS=`
(CaaS). The CaaS path uses the same model as the AIaaS path
(`Qwen/Qwen3-30B-A3B-Instruct-2507`); override with `--model`,
attention with `--attention {sdpa,flash_attention_2,eager}`, dtype
with `--dtype {bf16,fp16,fp32}`.

**CaaS hardware defaults.** `runai-submit-query-generate-local`
defaults to the **`h100` node pool** (target-specific override of
the `default` (a100) pool used by every other submit target).
Qwen3-30B-A3B in bf16 is ~61 GB of weights, so A100 40GB OOMs on
the weights alone; the `h100` pool on RCP carries H100 80GB + H200
141GB (per `make help`), both fit. The job does **not** pin a
specific GPU product — RCP's `restrict-nodename-runai-workloads`
policy rejects `--node-type` and requires `--node-pools` instead,
so node-pool routing is the right granularity.

The submit also exports
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so the caching
allocator releases fragmented blocks back to the pool between
batches — strongly recommended for any long-running local-LLM job
with variable-length inputs.

`QGL_BATCH_SIZE` is the `--batch-size` knob threaded into the CLI.
Defaults are calibrated against measured H100 80GB peak memory:

| Study | Doc tokens | `QGL_BATCH_SIZE` | Wallclock estimate |
| --- | --- | --- | --- |
| `study-A-fit` / `study-C-aggregator` | ≤ 8 192 | **4** (default) | ~1–2 h on study-A-fit (3600 jobs); `study-C-aggregator` reuses A-fit's queries via `impresso-research-study-seed` and skips re-generation. |
| `study-B-overflow` | ≥ 16 384 | **2** (override) | ~3–5 h |

At study-A-fit (8k-token docs, ~13k-token full prompts including
output budget), batch=8 OOMs at 78 GB. Batch=4 peaks ~70 GB with
~10 GB headroom. Bump UP only with VRAM telemetry; the OOM-aware
fallback inside the run loop will retry an OOMed batch as size-1
calls so a single bad batch doesn't lose its prompts, but
sustained OOMs eat wallclock and you should lower the default.

Override at submit time when H100 capacity is tight:

```bash
make STUDY=study-A-fit runai-submit-query-generate-local \
     RUNAI_NODE_POOL=default \
     QGL_BATCH_SIZE=2
```

(Only safe on the A100 80GB nodes within `default`; A100 40GB
OOMs.) Design rationale and rejected alternatives in
[`.progress/query-generation/notes.md`](./.progress/query-generation/notes.md).

Each scenario writes to a per-study, per-scenario S3 prefix:

```
s3://140-processed-data-sandbox/chunking-eval/<study>/embeddings/<id>_<label>/corpus.jsonl.bz2
```

so concurrent jobs and concurrent studies never collide. Job names
include the study (`embed-sweep-A-fit-s3`).

### Local smoke test

```bash
impresso-research-embed-sweep \
  --config configs/research/study-A-fit.yaml \
  --scenario S3 \
  --local-corpus tmp/chunking-eval/A-fit/corpus.jsonl.bz2 \
  --local-output tmp/S3.jsonl.bz2 \
  --no-upload
```

Per-flag CLI args (`--limit`, `--batch-size`, `--n-per-lg`, etc.)
override the YAML field-by-field for one-off experiments.

## Authoring a new study

```bash
cp configs/research/study-A-fit.yaml configs/research/study-mine.yaml
# edit study.name + the corpus/scenarios deltas
uv run python -m impresso_text_embedder.research.scenario_builder \
  --config configs/research/study-mine.yaml --list-table
```

The Pydantic loader rejects unknown keys, missing required fields,
incoherent token bounds, and path templates without a `{study}`
placeholder — typos fail at load time, not three hours into a sweep.
Single-level `extends:` only; chained inheritance is rejected.

## Output records

**Per-scenario shard** (`embed_sweep`, one JSONL line per content
item). Production text-level fields (`ci_id`, `model_id`,
`embedding`, `size`, `ts`, `ci_type`) plus research metadata
carried through from the manifest (`lg`, `year`, `provider`,
`alias`, `ocrqa`, `len_chars`) plus sweep annotations (`n_chunks`,
`scenario_id`, `chunker`, `chunk_tokens`) plus study provenance
(`study_name`, `study_config_sha`). Schema details in
`.progress/embedding-sweep/notes.md`.

**Queries-embedded shard** (`query_embed`, one JSONL line per
query). Production text-level primitives (`ci_id`, `model_id`,
`embedding`, `size`, `ts`) plus query identity (`query_id`) plus
the eval-relevant query metadata mirrored verbatim from
`queries.jsonl.bz2` (`lg`, `query_type`, `position_bucket`,
`position_chars`, `references`, `query_text`) plus the same
`study_name` / `study_config_sha` provenance. Schema details in
`.progress/query-embed/notes.md`.

## Logging

Sweep runs log a full INFO stream to
`/rcp-scratch/<user>/experiments/chunking-eval/<date>/<scenario_id>.log`
with a `tqdm` progress bar on the terminal (only `ERROR` records
echo, routed through `tqdm.write`). Outside RCP pass `--log-dir
<path>` — the CLI exits non-zero rather than falling back silently.

## Run on EPFL RCP / Run:AI

```bash
cp .env.docker.example .env.docker          # LDAP UID/GID, registry, project
make docker-build-push                      # linux/amd64 -> Harbor
make k8s-create-secrets                     # S3 + Harbor pull (idempotent)
make STUDY=study-A-fit runai-submit-research-all
```

The image tag defaults to the current git branch (slashes → dashes),
so this branch pushes to
`<registry>/<project>/<image>:research-chunking-eval`. `make help`
shows the resolved tag and every target.

Pick GPU / size pod:

```bash
make STUDY=study-A-fit runai-submit-research-all \
     RUNAI_NODE_POOL=h100 RUNAI_GPU_TYPE=NVIDIA-H100-80GB-HBM3 \
     RUNAI_CPU=8 RUNAI_MEMORY=32G
```

H200 lives in the `h100` pool — find labels with
`kubectl get nodes -L nvidia.com/gpu.product`. Same image runs on
A100 / H100 / H200; `accel.py` auto-detects.

Interactive debug pod:

```bash
make runai-interactive && make runai-bash
make runai-delete-debug    # tear down
```

## Repo map

- **`configs/research/`** — study YAMLs (`base.yaml` + per-study).
- **`src/impresso_text_embedder/research/`** — research CLIs +
  schema (`study_config.py`) + scenario builder.
- **`src/impresso_text_embedder/{embed,model,chunking,aggregation,...}`** —
  production embedder modules; the research path reuses them as a
  library and never touches their CLI surface.
- **`tests/`** — flat `test_<module>.py`; CUDA / xformers / S3 mocked
  at call sites so the suite runs CPU-only.
- **`.progress/<slug>/notes.md`** + **`.progress/plan.md`** — current
  research notes per step + live ledger.
- **`.history/`** — frozen migration-era design narrative.

## About

### Impresso

[Impresso — Media Monitoring of the Past](https://impresso-project.ch)
is an interdisciplinary research project that develops tools for
processing and exploring large media archives across modalities, time,
languages and national borders. Impresso 1 (2017–2021) was funded by
the SNSF under grant [CRSII5_173719](http://p3.snf.ch/project-173719);
Impresso 2 (2023–2027) by the SNSF under grant
[CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the
Luxembourg National Research Fund under grant 17498891.

### License

Provided as open source under the
[GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE)
v3 or later. Copyright (C) 2018–2024 The Impresso team.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>
