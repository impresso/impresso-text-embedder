# impresso-text-embedder

> **GPU-bound batch embedder** for [Impresso](https://impresso-project.ch) content items.
> Reads yearly `.jsonl.bz2` shards from S3, embeds them with
> [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
> via `sentence-transformers`, and writes the results back one-output-per-input.

| Script                    | Purpose                                                         |
| ------------------------- | --------------------------------------------------------------- |
| `impresso-embed-create`   | Walk a provider tree and embed every shard it finds.            |
| `impresso-embed-validate` | Structural check, or per-record cosine comparison + diagnostics. |


## Install

Requires **Python ≥ 3.10**.

### With [uv](https://docs.astral.sh/uv/) — recommended

Uses the committed lockfile for reproducible installs.

```bash
uv sync --extra dev
source .venv/bin/activate
```

### With pip / venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Credentials

Copy the template and fill in the blanks — the resulting `.env` is gitignored:

```bash
cp .env.example .env
```

| Variable                                          | Purpose                                 | Needed for                                 |
| ------------------------------------------------- | --------------------------------------- | ------------------------------------------ |
| `SE_ACCESS_KEY`, `SE_SECRET_KEY`, `SE_HOST_URL`   | S3 credentials (Switch Engines).        | Every CLI run.                             |
| `HARBOR_ROBOT_USERNAME`, `HARBOR_ROBOT_PASSWORD`  | Harbor registry robot account.          | Container push + Run:AI pull secret only.  |

## Create embeddings

### Minimal run

Embed every shard under a provider:

```bash
impresso-embed-create \
  --provider SNL \
  --input-bucket  22-rebuilt-final \
  --output-bucket 42-processed-data-final
```

### Inspect, then execute

List what *would* be processed:

```bash
impresso-embed-create --provider SNL \
  --alias EXP GDL --year-min 1910 --year-max 1920 \
  --limit 3 --dry-run
```

Then run for real:

```bash
impresso-embed-create --provider SNL \
  --alias EXP GDL --year-min 1910 --year-max 1920 \
  --batch-size 64
```

> [!TIP]
> Existing outputs are skipped when the input hasn't changed (S3 `LastModified` comparison).
> Pass `--force` to reprocess unconditionally.

### Argument defaults

Run `impresso-embed-create --help` for the flat list.

<details>
<summary>Grouped reference (click to expand)</summary>

#### Selection — what to process

| Flag                          | Default                       | Notes                                                                  |
| ----------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| `--provider`                  | *required*                    | Provider code, e.g. `SNL`.                                             |
| `--alias` *(repeatable)*      | *all aliases*                 | Filter to the listed aliases.                                          |
| `--year-min` / `--year-max`   | *no bound*                    | Skip shards outside this inclusive year range.                         |
| `--limit`                     | *none*                        | Process at most the first N shards (lex S3 order, before skip-check).  |
| `--input-bucket`              | `122-rebuilt-final`           | Holds `<provider>/<alias>/*.jsonl.bz2`.                                |
| `--output-bucket`             | `140-processed-data-sandbox`  | Outputs mirror input under `embeddings/docs/<model-slug>/`.            |
| `--input-prefix`              | `""`                          | Optional prefix inside the input bucket.                               |

#### Model & output shape

| Flag                  | Default                              | Notes                                              |
| --------------------- | ------------------------------------ | -------------------------------------------------- |
| `--model-name`        | `Alibaba-NLP/gte-multilingual-base`  | HuggingFace model id.                              |
| `--model-revision`    | `f7d567e`                            | HF commit pin; not appended to the output slug.    |
| `--embedding-level`   | `text`                               | `text` / `sentence` / `chunk`.                     |
| `--chunking-strategy` | `semantic`                           | Used only when `--embedding-level=chunk`.          |

#### Record filtering

| Flag                | Default | Notes                                                                |
| ------------------- | ------- | -------------------------------------------------------------------- |
| `--min-char-length` | `800`   | Skip records whose reconstructed text is shorter.                    |
| `--content-type`    | `ar`    | Keep only records whose `tp` is in this allow-list (`ar` / `page`).  |

#### Long-document handling

| Flag                      | Default                       | Notes                                                                                        |
| ------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| `--long-doc-strategy`     | `chunk`                       | `chunk` (split + aggregate) or `truncate` (legacy pre-step-16).                              |
| `--long-doc-chunk-tokens` | *auto, tokenizer-derived*     | `model_max_length − num_special_tokens_to_add(pair=False)` (8190 for gte-multilingual-base). |
| `--long-doc-aggregation`  | `mean`                        | Only choice today.                                                                           |

#### Performance / GPU

| Flag                                            | Default                       | Notes                                                                                        |
| ----------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| `--batch-size`                                  | *auto, per GPU profile*       | A100=64, H100/H200=128, other CUDA=32, CPU=8.                                                |
| `--precision`, `--attention`, `--unpad-inputs`  | bf16 / xformers / on          | Numerical-ablation levers — see [Numerical-ablation toggles](#numerical-ablation-toggles).   |

#### Sharding & run modes

| Flag                            | Default | Notes                                                                                       |
| ------------------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `--shard-index` / `--num-shards`| `0` / `1` | Round-robin partition over `list_objects_v2` lex order. See [Multi-shard runs](#multi-shard-runs-horizontal-throughput). |
| `--force`                       | `False` | Reprocess even if output exists on S3.                                                      |
| `--dry-run`                     | `False` | List files; no model load, no writes.                                                       |

#### Logging

| Flag                | Default                                       | Notes                                                                                       |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `--log-level-file`  | `INFO`                                        | `DEBUG` / `INFO` / `WARNING` / `ERROR`.                                                     |
| `--log-dir`         | `/rcp-scratch/<user>/experiments/embeddings`  | Full path: `<log-dir>/<YYYY-MM-DD>/<provider>.log` (or `<provider>-shard-<i>-of-<N>.log` when `--num-shards` > 1). |

</details>

### Embedding levels

`--embedding-level` picks **what** gets embedded. It is independent of the
chunking registry, which only applies at `chunk` level.

| Level                | Produces                                    | Long-text handling                                                                                                       |
| -------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `text` *(default)*   | One vector per content item.                | Docs > 8192 tokens are chunked (`fixed-window`) and mean-pooled into one vector. Opt out with `--long-doc-strategy truncate`. |
| `sentence`           | One vector per pre-tokenised `sents` entry. | No chunker runs.                                                                                                         |
| `chunk`              | One vector per chunk.                       | `--chunking-strategy` selects `semantic` (default) or `token-budget`.                                                    |

> [!NOTE]
> `text` and `sentence` are *embedding levels*, not chunking strategies —
> they do not appear in the chunking registry.

### Numerical-ablation toggles

Three flags expose the post-migration fast-path levers. Defaults preserve
the historical fast path — running with no overrides reproduces today's
outputs — but each can be flipped independently to bisect drift against
an older baseline.

| Flag                                  | Default     | What it controls                                                                                                                                                           | Reference                                                                                                                                                                     |
| ------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--precision {bf16,fp32}`             | `bf16`      | Whether `model.encode()` runs inside a `torch.autocast(dtype=bfloat16)` scope on CUDA. `fp32` skips the autocast; model weights are fp32 either way.                       | [PyTorch blog — What every user should know about mixed precision training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) |
| `--attention {xformers,eager}`        | `xformers`  | Whether `use_memory_efficient_attention=True` is set on the model config. Alibaba's modeling file then routes attention through `xformers.ops.memory_efficient_attention` (which dispatches to FA2 on Ampere, FA3 on Hopper). `eager` omits the flag and uses the model's default attention path. | [HuggingFace — Flash Attention (concept)](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)                                                    |
| `--unpad-inputs` / `--no-unpad-inputs`| `True`      | Whether `unpad_inputs=True` is set on the model config. The modeling file then strips padding tokens before attention so variable-length sequences don't waste FLOPs on PADs. | [HF blog — Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2)                                                                                      |

> [!TIP]
> Typical bisection grid against a legacy baseline: defaults / `--precision fp32` /
> `--attention eager` / all three flipped. Whichever combination collapses
> drift to ~zero identifies the responsible lever. `--attention=xformers`
> with no CUDA device or no importable `xformers` package fails fast at model
> load — no silent fallback.

## Validate an output file

### Structural only

Schema shape, dimensionality, absence of NaNs:

```bash
impresso-embed-validate s3://<bucket>/.../EXP-1912.jsonl.bz2
```

### Against a reference

Per-record cosine comparison (default tolerance `1e-4`):

```bash
impresso-embed-validate \
  s3://<bucket>/.../EXP-1912.jsonl.bz2 \
  --target s3://<bucket>/golden/EXP-1912.jsonl.bz2
```

### With drift diagnostics

Add `--source <input.jsonl.bz2>` to surface char-length histograms,
`lg` / `tp` breakdowns, and worst-drift excerpts for records that mismatched
or went missing:

```bash
impresso-embed-validate \
  s3://<bucket>/.../EXP-1912.jsonl.bz2 \
  --target s3://<bucket>/golden/EXP-1912.jsonl.bz2 \
  --source s3://<bucket>/inputs/EXP-1912.jsonl.bz2
```

See the *"Validate — source-backed diagnostics"* decision in
[`CLAUDE.md`](./CLAUDE.md) for the full output format.

### Export to CSV

`--csv-out <dir>` dumps mismatches as spreadsheet-friendly CSVs alongside the Rich panels:

```bash
impresso-embed-validate s3://.../EXP-1912.jsonl.bz2 \
  --target s3://.../golden/EXP-1912.jsonl.bz2 \
  --source s3://.../inputs/EXP-1912.jsonl.bz2 \
  --csv-out ./out
```

| File                  | One row per                              | Columns                                                |
| --------------------- | ---------------------------------------- | ------------------------------------------------------ |
| `above_threshold.csv` | record whose cosine distance > `--tol`   | `ci_id, url, distance, lg, tp, char_length`            |
| `missing.csv`         | record present on only one side          | `ci_id, url, direction, lg, tp, char_length`           |

`url` links to the Impresso web app. `lg` / `tp` / `char_length` are populated only with `--source`; sentence/chunk runs append `item_id, id_key`. Exit code is `0` on pass, non-zero on failure.

## Logging

| Destination | Contents                                                                                             | Controlled by                   |
| ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------- |
| Terminal    | Single `tqdm` bar with per-file postfix `dl=…s enc=…s up=…s gpu=…%`.                                 | (always on)                     |
| Disk        | Full `INFO` log at `/rcp-scratch/<user>/experiments/embeddings/<YYYY-MM-DD>/<provider>.log`.         | `--log-dir`, `--log-level-file` |

Outside RCP, `--log-dir` is required — the CLI exits non-zero rather than
falling back silently.

## Development

```bash
pytest
ruff check .
```

Repo map:

- **`src/impresso_text_embedder/`** — package source.
- **`tests/`** — unit + integration tests.
- **`.progress/<slug>/notes.md`** — design notes per subsystem.

## Run on EPFL RCP / Run:AI

```bash
cp .env.docker.example .env.docker       # LDAP UID/GID, registry, project
make docker-build-push                   # build linux/amd64 → Harbor
make k8s-create-secrets                  # S3 + Harbor pull secret (idempotent)
make runai-submit PROVIDER=BNL \
     INPUT_BUCKET=22-rebuilt-final \
     OUTPUT_BUCKET=42-processed-data-final \
     EMBED_EXTRA_ARGS="--embedding-level text --batch-size 64"
```

Pass these as **make variables** (`NAME=value`), not CLI flags (`--name=value`) —
`make` parses anything starting with `--` as one of its own options and rejects it.
CLI flags for `impresso-embed-create` go inside `EMBED_EXTRA_ARGS="…"`.

`make help` lists every target. See [`.progress/docker-runai/`](./.progress/docker-runai/)
for setup rationale and the secret conventions.

### Picking the GPU and sizing the pod

`RUNAI_NODE_POOL` picks the pool (`default` = A100, `h100`, `v100`).
`RUNAI_GPU_TYPE` pins the GPU product *within* the pool — the only way to
split H100 vs H200, since `--node-pools` doesn't distinguish them. Find the
labels on RCP with `kubectl get nodes -L nvidia.com/gpu.product`. Same image
runs on all three; `accel.py` auto-detects the arch at startup.

`RUNAI_CPU` / `RUNAI_MEMORY` (with optional `RUNAI_CPU_LIMIT` /
`RUNAI_MEMORY_LIMIT`) request resources explicitly so the 10-way S3
prefetch + encode + upload overlap isn't starved by noisy neighbours.
A reasonable starting point is `RUNAI_CPU=8 RUNAI_MEMORY=32G`; tune from
the per-file `dl=…s enc=…s up=…s` postfix on the `tqdm` bar. Unset →
namespace default applies.

```bash
make runai-submit PROVIDER=BNL \
     INPUT_BUCKET=22-rebuilt-final \
     OUTPUT_BUCKET=42-processed-data-final \
     RUNAI_NODE_POOL=h100 RUNAI_GPU_TYPE=NVIDIA-H200-141GB \
     RUNAI_CPU=8 RUNAI_MEMORY=32G
```

### Multi-shard runs (horizontal throughput)

Each Run:AI job owns one GPU. To process a provider in parallel across N
GPUs, submit N independent jobs with disjoint file partitions:

```bash
make runai-submit-multi PROVIDER=BNL NUM_SHARDS=4 \
     INPUT_BUCKET=22-rebuilt-final \
     OUTPUT_BUCKET=42-processed-data-final \
     EMBED_EXTRA_ARGS="--embedding-level text --batch-size 64"
```

`EMBED_EXTRA_ARGS` is forwarded to every shard so flags apply uniformly.
This loops `i` from 0 to N-1 and submits a separate job for each shard.
Each job runs `impresso-embed-create --shard-index i --num-shards N` and
processes a round-robin partition of the file list (over
`list_objects_v2`'s lexicographic order — deterministic, no coordination).
Failed shards re-run idempotently via the existing
`--skip-if-s3-exists` + input-newer-than-output check; just resubmit the
same shard. Per-shard logs land at
`<log-dir>/<YYYY-MM-DD>/<provider>-shard-<i>-of-<N>.log` so concurrent
shards don't stomp each other.

`make runai-submit-shard PROVIDER=BNL SHARD_INDEX=2 NUM_SHARDS=4` submits
just one shard (handy for re-running a single failed shard). Defaults
`SHARD_INDEX=0 NUM_SHARDS=1` reproduce the unsharded `runai-submit`
behaviour. Design rationale and rejected alternatives in
[`.progress/multi-gpu-sharding/`](./.progress/multi-gpu-sharding/).

### Interactive debug shell

For PVC / GPU / model-cache checks, or to run the CLI by hand:

```bash
make runai-interactive    # submit the debug pod
make runai-bash           # shell in once it's Running
make runai-delete-debug   # tear it down when done
```

Override `RUNAI_DEBUG_JOB_NAME` / `RUNAI_DEBUG_GPU` via `make` to run several in parallel.

---

## About

### Impresso

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an
interdisciplinary research project that aims to develop and consolidate tools for
processing and exploring large collections of media archives across modalities, time,
languages and national borders. The first project (2017-2021) was funded by the Swiss
National Science Foundation under grant
No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027)
by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585))
and the Luxembourg National Research Fund under grant No. 17498891.

### Copyrights

Copyright (C) 2018-2024 The Impresso team.  
Contributors to this program include: [Simon Clematide](https://github.com/simon-clematide)

### License

This program is provided as open source under
the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE)
v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>