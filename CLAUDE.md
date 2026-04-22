# impresso-text-embedder

Python package that computes multilingual text embeddings for the [Impresso](https://impresso-project.ch) project over large yearly `.jsonl.bz2` shards stored on S3.

This file is the orientation document for Claude. If something here is wrong or out of date, fix it here first — future sessions rely on it.

---

## Status

**The repo is mid-migration.** The previous Make/script-based layout (`lib/*.py`, `Makefile`, `run.sh`, `Pipfile`, `requirements.txt`, …) has been deleted on branch `feat/migration-python-package`. Nothing Python is present yet besides `.venv/`. The target is a proper `pyproject.toml`-based package.

The reference for behavior is the **old code on `main`** — most notably `lib/text_embedding_processor.py` (schemas, chunking logic, S3 conventions). Read it with `git show main:lib/text_embedding_processor.py` when you need ground truth. Do not treat it as code to port line-for-line; treat it as a spec.

---

## Goal

A single Python package, `impresso-text-embedder`, that:

1. Embeds `.jsonl.bz2` files of Impresso content items using [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) via `sentence-transformers`.
2. Exposes **two CLI scripts**:
   - **Create** — takes a `--provider`, walks all aliases/years under that provider, produces one output `.jsonl.bz2` per input file.
   - **Validate** — takes a direct path to one `.jsonl.bz2` and verifies it matches a reference embedding file (if present) under a numerical tolerance.
3. Runs **GPU-bound**. CPU/IO must never be the bottleneck.

---

## Target hardware (for now)

**A100 only.** Single GPU. Applied today (`model.py`):
- `bfloat16` via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around `model.encode`. Model weights stay fp32 so LayerNorm stays stable. No `.to(bfloat16)` on the module.
- `torch.inference_mode()` around encoding.
- `trust_remote_code=True` (required by `gte-multilingual-base`).

Deferred / opt-in (decide when we can measure on real A100 — see `.progress/gpu-throughput/notes.md`):
- **xformers + unpadding** — the documented acceleration path for this model family. To be added as an `accel` extra in `pyproject.toml` with the matching `model_kwargs` passthrough.
- `flash-attn` v2 — **not** documented as supported by `gte-multilingual-base`. Don't wire unless xformers is insufficient.
- `torch.compile` — benchmark before enabling; ST's encode path may not trace cleanly.
- Avoid fp16 unless bf16 is shown to underperform — the point of A100 is bf16.

Multi-GPU, older TITAN, or newer Hopper/Blackwell variants are **explicitly out of scope** right now. The `pyproject.toml` should be structured so variants can be added later (e.g. as optional dependency groups or extras), but do not over-engineer that now.

---

## S3 layout

Input (one bucket, one file per newspaper-year):
```
s3://<input-bucket>/<provider>/<alias>/<alias>-<year>.jsonl.bz2
```

Output (mirrored 1:1):
```
s3://<output-bucket>/embeddings/docs/<model-slug>/<provider>/<alias>/<alias>-<year>.jsonl.bz2
```

- `<model-slug>` derives from the HF model name — use `gte-multilingual-base` (drop the `Alibaba-NLP/` prefix). Revision pinning is a CLI flag; include it in the slug only if it changes the output (decide at implementation time, document the choice here when you do).
- **1 input file → 1 output file, same basename.** No sharding, no merging.
- Some provider/alias/year combinations are large, some tiny. Do not assume uniform size.
- Skip processing when the output already exists on S3 (old behavior: `--quit-if-s3-output-exists`). Expose this as the default and an opt-out flag.

Credentials live in `.env` at the repo root. **Never read the contents of `.env`.** The variable names expected by the old code are `SE_ACCESS_KEY`, `SE_SECRET_KEY`, `SE_HOST_URL` — use these names and let the user fill them.

---

## Data contract

All schema details below are taken from `main:lib/text_embedding_processor.py`. If you need more detail than this summary, read that file.

### Input record (one per line in the input `.jsonl.bz2`)
- `id` — canonical content-item id (string).
- `tp` — content type; filter to `ar` (articles) by default, allow `page` via CLI.
- `lg` — language code (optional).
- `sents` — list of sentence dicts. Each sentence has `tok: [{t: <token>, o: <char-offset>}, …]` and an optional sentence-level offset `o`.
- `ft` may be absent; the full text is reconstructed from `sents` via `rebuild_ft_from_offsets`. Preserve that helper's exact behavior (offsets are absolute, gaps are padded with spaces).
- `lingproc_path` — optional provenance path that must be echoed through to the output when present.

### Output records
One line per content item. Shape depends on the embedding level:
- **text** (default, one embedding per document): flat record `{id, ts, embedder, len, embedding, text?}`.
- **sentence**: `{ts, ci_id, sents: [{sent_id, embedding, size, lg?, o?}, …], model_id?, lingproc_path?, git?}`.
- **chunk**: `{ts, ci_id, chunks: [{chunk_id, embedding, size, lg?, o?}, …], model_id?, lingproc_path?, git?}`.

Embeddings are stored as a list of floats rounded to 5 decimals. `ts` is UTC `YYYY-MM-DDTHH:MM:SSZ`.

### Filtering
- `--min-char-length` gates very short texts (old default: 400 for text, effectively the same for chunks; sentence-level used the same threshold). Items below the threshold are counted and skipped, not errored.

### Chunking
The model's max sequence is 8192 tokens. For inputs longer than that:
- **First strategy to implement:** port the existing semantic chunker (`chonkie.SemanticChunker` with `minishlab/potion-base-8M`, threshold 0.5, chunk_size 1024, min_sentences 5) — that's what the old code does.
- **Future:** sliding-window, sentence-group, token-budget. Structure the code so strategies are pluggable (e.g. a registry keyed by a CLI string), but don't build more than one strategy until it's actually needed. Three similar lines beats a premature abstraction.

---

## CLIs

Script entry points live in `[project.scripts]`. Proposed names (rename if something cleaner emerges):

### `impresso-embed-create`
- Required: `--provider <CODE>` (e.g. `SNL`).
- Walks `s3://<input-bucket>/<provider>/` and processes every `<alias>-<year>.jsonl.bz2` found. Filter flags (`--alias`, `--year-from`, `--year-to`) optional; add only when needed.
- Other flags, carried over or adapted from the old CLI: `--model-name`, `--model-revision`, `--embedding-level {text,sentence,chunk}`, `--batch-size`, `--min-char-length`, `--normalize-embeddings`, `--content-type {ar,page}`, `--include-text`, `--no-overwrite`, `--skip-if-s3-exists` (default on), `--dry-run`.
- Expose `--batch-size` explicitly. The old code hardcoded `batch_size=8`, which is not A100-appropriate — tune based on GPU headroom and document the chosen default here once measured.

### `impresso-embed-validate`
- Positional: path to a single `.jsonl.bz2` (S3 URI or local).
- Optional: `--target <path>` — reference embeddings file to compare against. If omitted, do structural validation only (schema shape, no NaNs, expected dimensionality).
- `--tol <float>` — numerical tolerance. Default and metric (cosine distance vs. L∞) should be picked and documented here when implemented. Lean toward **cosine distance ≤ tol** for normalized vectors, L∞ otherwise.
- Exits non-zero on failure; prints a summary of mismatches.

---

## Throughput: keeping the GPU saturated

The `.jsonl.bz2` shards can be large (minutes of IO per file). The previous code was naive: read the whole file, then encode. The new code should overlap IO/decompression/tokenization with GPU encode. Concretely:

- **Stream, don't slurp.** Read lines from S3 with `bz2.open()` around an `smart_open`/fsspec stream; do not load the entire file into memory.
- **Prefetch ahead.** While file N is being encoded on the GPU, file N+1 should already be downloading/decompressing on a background thread.
- **Batch across records.** `sentence_transformers.encode(..., batch_size=B)` with a large B (tune on A100; start at 64 for full documents, much higher for sentences/chunks). Pass `convert_to_numpy=True`.
- **Upload async.** Upload file N's output to S3 while starting file N+1, not after.
- **Measure.** Keep a simple `nvidia-smi dmon`-style GPU-util log or emit periodic util stats; the acceptance bar is sustained >85% SM utilization on A100 during steady state.

Don't invent a complex multi-process pipeline. A single process with a small thread pool (one prefetcher, one uploader) plus the GPU worker is almost certainly enough. Revisit only if measurement says otherwise.

---

## Reuse `impresso-essentials`

The shared Impresso utilities live at <https://github.com/impresso/impresso-essentials> (docs: <https://impresso.github.io/impresso-essentials/_build/html/index.html>). Prefer these over rolling our own S3/IO code:

- `impresso_essentials.io.s3` — `get_s3_client`, `get_s3_resource`, `get_bucket`, `get_storage_options`, `upload_to_s3`, `list_s3_directories`, `list_providers_and_aliases`, `fixed_s3fs_glob`, `s3_glob_with_size`, `extract_provider_alias_key`, `provider_in_path`. Prefer these over hand-rolling `boto3.resource("s3", ...)` / `smart_open` wrappers.
- `impresso_essentials.io.fs_utils` — local FS helpers.
- `impresso_essentials.text_utils` — text processing.
- `impresso_essentials.versioning.*` — data manifests. Relevant if we start emitting a manifest for the embedding outputs (not an immediate requirement).

**Two deliberate exceptions** (see `.progress/io-layer/notes.md`):

- `read_jsonlines` is **not** used for the encoder hot path — it does `body.read()` + `bz2.decompress(data)` and loads the full file into memory, which breaks the streaming/prefetch model. Our `io.iter_jsonl_bz2` opens the S3 body stream and wraps it with `bz2.open` for chunked decoding.
- `upload_to_s3` returns `bool` and swallows exceptions; our `io.upload_local_file` wraps it and raises on `False`, so a failed upload can't masquerade as a persisted output.

`impresso-essentials` is pinned at `>=1.4.1` in `pyproject.toml`.

---

## Project layout (target)

```
impresso-text-embedder/
├── pyproject.toml              # hatchling build, uv-friendly
├── CLAUDE.md                   # this file
├── README.md                   # user-facing docs (keep short; link back here)
├── LICENSE                     # AGPL-3.0 (matches main README)
├── .env                        # local only, gitignored — do not read
├── src/
│   └── impresso_text_embedder/
│       ├── __init__.py
│       ├── cli/
│       │   ├── create.py       # impresso-embed-create
│       │   └── validate.py     # impresso-embed-validate
│       ├── embed.py            # model loading + encode loop
│       ├── chunking/           # pluggable strategies
│       │   ├── base.py
│       │   └── semantic.py     # chonkie-based (initial)
│       ├── io.py               # thin wrappers around impresso_essentials.io.s3
│       ├── schema.py           # input/output record types
│       └── text.py             # rebuild_ft_from_offsets, rebuild_sentence_from_offsets
└── tests/
    ├── fixtures/               # tiny .jsonl.bz2 for integration
    └── test_*.py
```

Import name uses underscores: `impresso_text_embedder`. Distribution name uses hyphens: `impresso-text-embedder`.

---

## Tooling conventions

- **Build backend:** hatchling. **Dependency manager:** uv (fast, and `uv sync`/`uv lock` handle the lockfile cleanly). Use `uv run <cmd>` in examples.
- **Python:** `>=3.10` (old code uses PEP 604 `|` unions).
- **Lint/format:** ruff (drop the old `.flake8`).
- **Type check:** optional for now; prefer type hints on public functions, skip mypy unless it earns its keep.
- **Tests:** pytest. Unit tests for pure-Python helpers (`rebuild_ft_from_offsets`, chunkers, schema). One integration test that runs `impresso-embed-create` against a tiny local fixture end-to-end (no S3). No pre-commit hooks.
- **License:** AGPL-3.0-or-later (README on `main` is authoritative).
- **Env loading:** `python-dotenv` at CLI entry. Don't eagerly load S3 clients at import time.

---

## Commands (once pyproject exists)

```bash
uv sync                                   # install
uv run impresso-embed-create --provider SNL --embedding-level text --batch-size 128
uv run impresso-embed-validate s3://.../EXP-1912.jsonl.bz2 --target s3://.../golden/EXP-1912.jsonl.bz2 --tol 1e-4
uv run pytest
uv run ruff check .
```

---

## Non-goals (explicit)

- Multi-GPU, multi-node, DDP.
- Non-A100 GPU variants (deferred).
- Incremental/partial-file recovery mid-shard. A file either completes or is redone.
- Local-only workflows beyond tests — production flow is S3 in, S3 out.
- A new Makefile or stamp-based orchestration. The `main`-branch Make layer is being deliberately replaced by a Python CLI.

---

## Things to decide before / while implementing

Leave these as open questions; don't paper over them:

- Default `--batch-size` per embedding level on A100 (measure, don't guess).
- Validation metric + default tolerance.
- Whether to include the model revision hash in `<model-slug>` or keep it separate.
- Whether to emit an impresso-essentials data manifest alongside outputs.
- Exact output JSON encoding choice (compact vs. readable; old code is inconsistent between levels — pick one, document here).
