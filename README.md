# impresso-text-embedder

> **GPU-bound batch embedder** for [Impresso](https://impresso-project.ch) content items.
> Reads yearly `.jsonl.bz2` shards from S3, embeds them with
> [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
> via `sentence-transformers`, and writes the results back one-output-per-input.

| Script                    | Purpose                                                         |
| ------------------------- | --------------------------------------------------------------- |
| `impresso-embed-create`   | Walk a provider tree and embed every shard it finds.            |
| `impresso-embed-validate` | Structural check, or per-record cosine comparison + diagnostics. |

**Looking for deeper detail?**

- [`CLAUDE.md`](./CLAUDE.md) — data contract, S3 layout, hardware profile, full list of design decisions.
- [`.progress/`](./.progress/) — per-subsystem notes (I/O, chunking, GPU tuning, validation, …).
- [`.progress/plan.md`](./.progress/plan.md) — migration status.

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

| Flag                      | Default                                         | Notes                                                                 |
| ------------------------- | ----------------------------------------------- | --------------------------------------------------------------------- |
| `--provider`              | *(required)*                                    | Provider code, e.g. `SNL`.                                            |
| `--input-bucket`          | `122-rebuilt-final`                             | S3 bucket holding `<provider>/<alias>/*.jsonl.bz2`.                   |
| `--output-bucket`         | `140-processed-data-sandbox`                    | Outputs mirror input under `embeddings/docs/<model-slug>/`.           |
| `--input-prefix`          | `""`                                            | Optional prefix inside the input bucket.                              |
| `--model-name`            | `Alibaba-NLP/gte-multilingual-base`             | HuggingFace model id.                                                 |
| `--model-revision`        | `f7d567e`                                       | HF commit pin; not appended to the output slug.                       |
| `--embedding-level`       | `text`                                          | `text` / `sentence` / `chunk`.                                        |
| `--chunking-strategy`     | `semantic`                                      | Only used when `--embedding-level=chunk`.                             |
| `--batch-size`            | *(auto, per GPU profile)*                       | A100=64, H100/H200=128, other CUDA=32, CPU=8.                         |
| `--min-char-length`       | `400`                                           | Records with shorter reconstructed text are skipped.                  |
| `--content-type`          | `ar`                                            | Keep only records whose `tp` is in this allow-list (`ar` / `page`).   |
| `--long-doc-strategy`     | `chunk`                                         | `chunk` (split + aggregate) or `truncate` (legacy pre-step-16).       |
| `--long-doc-chunk-tokens` | *(auto, tokenizer-derived)*                     | `model_max_length − num_special_tokens_to_add(pair=False)` (8190 for gte-multilingual-base). |
| `--long-doc-aggregation`  | `mean`                                          | Only choice today.                                                    |
| `--alias`                 | *(none — all aliases)*                          | Filter to the listed aliases.                                         |
| `--year-min`              | *(none)*                                        | Skip shards whose year is strictly below this.                        |
| `--year-max`              | *(none)*                                        | Skip shards whose year is strictly above this.                        |
| `--force`                 | `False`                                         | Reprocess even if output exists on S3.                                |
| `--dry-run`               | `False`                                         | List files without loading the model or writing outputs.              |
| `--limit`                 | *(none)*                                        | Process at most the first N listed shards (applied before skip-check). |
| `--log-level-file`        | `INFO`                                          | `DEBUG` / `INFO` / `WARNING` / `ERROR`.                               |
| `--log-dir`               | `/rcp-scratch/<user>/experiments/embeddings`    | Full path becomes `<log-dir>/<YYYY-MM-DD>/<provider>.log`.            |

Run `impresso-embed-create --help` for per-flag descriptions.

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

Add `--csv-out <dir>` to dump two machine-readable lists alongside the
Rich panels — useful when triaging many mismatches in a spreadsheet or
piping them to follow-up tooling:

```bash
impresso-embed-validate \
  s3://<bucket>/.../EXP-1912.jsonl.bz2 \
  --target s3://<bucket>/golden/EXP-1912.jsonl.bz2 \
  --source s3://<bucket>/inputs/EXP-1912.jsonl.bz2 \
  --csv-out ./out
```

This writes:

- `./out/above_threshold.csv` — one row per record whose cosine
  distance exceeded `--tol`. Columns: `ci_id, url, distance, lg, tp,
  char_length`.
- `./out/missing.csv` — one row per record present on only one side.
  Columns: `ci_id, url, direction, lg, tp, char_length`. The
  `direction` column is `missing_in_target` or `missing_in_produced`.

For sentence/chunk-level runs, `item_id` and `id_key` are appended at
the end of each row so per-item drifts can still be triaged. The `url`
column links each row to the Impresso web app
(`https://impresso-project.ch/app/article/{ci_id}`) so reviewers can
open the article in one click. Source-derived columns (`lg`, `tp`,
`char_length`) are populated only when `--source` is also supplied;
otherwise they are left blank. Aggregate diagnostics
(`reconstructable` / `empty` / `below_min_char` counts, char-length
distribution, language and content-type breakdowns) live in the
source-stats panel printed to stdout, not in the CSV rows.

Exit code is `0` on pass, non-zero on failure.

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

`make help` lists every target. See [`.progress/docker-runai/`](./.progress/docker-runai/)
for setup rationale and the secret conventions.

### Interactive debug shell

For PVC / GPU / model-cache checks, or to run the CLI by hand:

```bash
make runai-interactive    # submit the debug pod
make runai-bash           # shell in once it's Running
make runai-delete-debug   # tear it down when done
```

Override `RUNAI_DEBUG_JOB_NAME` / `RUNAI_DEBUG_GPU` via `make` to run several in parallel.

## License

[AGPL-3.0-or-later](./LICENSE).
