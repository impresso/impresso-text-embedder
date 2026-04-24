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

## Target hardware

**A100 and H100 (incl H200), single GPU, same image, same code path.** The
encoder auto-detects the arch at startup (see `accel.py`) and picks the
right default batch size; xformers' `memory_efficient_attention` picks the
FA2 (Ampere) or FA3 (Hopper) kernel transparently from there.

Applied today (`model.py` + `accel.py`):
- `bfloat16` via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around `model.encode`. Model weights stay fp32 so LayerNorm stays stable. No `.to(bfloat16)` on the module. Inside the autocast region Q/K/V are bf16, which is what xformers needs to fire the FA kernel.
- `torch.inference_mode()` around encoding.
- `trust_remote_code=True` (required by `gte-multilingual-base`).
- `config_kwargs={"unpad_inputs": True, "use_memory_efficient_attention": True}` passed to `SentenceTransformer` when xformers is importable and the device is CUDA. These land on the model config (where Alibaba's custom modeling file reads them from) — **not** `model_kwargs`; sentence-transformers v5 pre-loads the config and passes it to `from_pretrained` explicitly, which skips HF's kwarg-to-config routing and would make `NewModel.__init__` reject them. HF-generic `attn_implementation` is **rejected** by the model anyway (`NewModel does not support Flash Attention 2.0 yet`). xformers is not in NGC 25.03 — the Dockerfile installs it explicitly from PyTorch's `cu128` wheel index with `--no-deps` so NGC's torch stays untouched. Full research in `.progress/gpu-profiles/notes.md`.
- Per-profile `--batch-size` default: A100=64, H100/H200=128, unknown-CUDA=32, CPU=8. Explicit `--batch-size N` overrides.

Deferred levers (not wired, decide on measurement — see `.progress/gpu-profiles/notes.md` and `.progress/gpu-throughput/notes.md`):
- **FP8 via TransformerEngine** on Hopper — potential ~2× on top of bf16, but requires TE or TensorRT-LLM integration. Real engineering project. TE 2.1 is already in NGC 25.03, so no new dependency — just wiring.
- **`flash-attn` alongside xformers** — meaningful H100 lever. PyPI xformers wheels ship only a CUTLASS `memory_efficient_attention` kernel; for the Hopper-specialised FA3 kernel (and Ampere FA2) you need the separate `flash-attn` package installed too. Today the Dockerfile installs xformers only (CUTLASS path) — fine on A100, leaves some H100 headroom on the table. Adding `flash-attn` is a Dockerfile line + a prebuilt-wheel URL, gated on live-GPU measurement showing FA3 isn't firing.
- `torch.compile` — benchmark before enabling; ST's encode path may not trace cleanly.
- Avoid fp16 unless bf16 is shown to underperform — the point of bf16 is same-range-as-fp32 without loss scaling.

Multi-GPU / DDP, V100 (no bf16 tensor cores), and Blackwell variants are **out of scope** today. Adding Blackwell later is mostly a new profile entry in `accel.py`.

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

- `<model-slug>` follows the Impresso versioned-embedding convention. For the shipped model (`Alibaba-NLP/gte-multilingual-base`) the slug is **`embeddings_gte_v1-1-0`**. Unknown models fall back to stripping the HF vendor prefix. The override map lives in `pipeline.MODEL_SLUG_OVERRIDES`; add an entry there when a new Impresso-blessed model lands. Revision pinning is a CLI flag; it is **not** appended to the slug (different revisions of the same model go to the same output path).
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
- **text** (default, one embedding per document) — follows the authoritative Impresso document-embeddings schema (`embeddings-docs.schema.json`): required `{ci_id, model_id, embedding, size}`; optional `{ts, ci_type}`.
- **sentence**: `{ts, ci_id, sents: [{sent_id, embedding, size, lg?, o?}, …], model_id?, lingproc_path?, git?}`.
- **chunk**: `{ts, ci_id, chunks: [{chunk_id, embedding, size, lg?, o?}, …], model_id?, lingproc_path?, git?}`.

Embeddings are stored as a list of floats rounded to 5 decimals. `ts` is UTC `YYYY-MM-DDTHH:MM:SSZ`. Sentence- and chunk-level schemas are not yet re-verified against authoritative specs; the text-level schema has been.

### Filtering
- `--min-char-length` gates very short texts (old default: 400 for text, effectively the same for chunks; sentence-level used the same threshold). Items below the threshold are counted and skipped, not errored.
- `--content-type` rejects records whose `tp` is missing **or** not in the allow-list. Missing `tp` is tallied as `missing_content_type` (distinct from `content_type` for the allow-list reject) and emits **one WARNING per file** on first occurrence so a malformed shard surfaces immediately. Every filter reason is surfaced in the per-file `done` INFO line as `skipped=<total> (reason1=<n1> …)`. Full reference of every filter key + trigger + log level in `.progress/record-filtering/notes.md`.

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
- Other flags, carried over or adapted from the old CLI: `--model-name`, `--model-revision`, `--embedding-level {text,sentence,chunk}`, `--batch-size`, `--min-char-length`, `--normalize-embeddings`, `--content-type {ar,page}`, `--no-overwrite`, `--skip-if-s3-exists` (default on), `--dry-run`.
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
- **Batch across records.** `sentence_transformers.encode(..., batch_size=B)` with a large B (A100: start at 64 for full documents, H100: 128+; sentences/chunks can go much higher). Pass `convert_to_numpy=True`.
- **Upload async.** Upload file N's output to S3 while starting file N+1, not after.
- **Measure.** Keep a simple `nvidia-smi dmon`-style GPU-util log or emit periodic util stats; the acceptance bar is sustained >85% SM utilization during steady state (on whichever arch the job lands on).

Don't invent a complex multi-process pipeline. A single process with a small thread pool (one prefetcher, one uploader) plus the GPU worker is almost certainly enough. Revisit only if measurement says otherwise.

---

## `impresso-essentials` — vendored, not imported

The shared Impresso utilities live at <https://github.com/impresso/impresso-essentials> (docs: <https://impresso.github.io/impresso-essentials/_build/html/index.html>). We originally imported its S3 helpers, but we **do not depend on the package** — it is not in `pyproject.toml` and not installed in the Docker image. Reason: `impresso-essentials==1.4.x` hard-pins `numpy==2.2.1` (and dask/pandas/pyarrow…) in its distribution metadata, which would uninstall NGC's `numpy==1.26.4` and break the apex/NCCL/transformer-engine/xformers ABI stack. A `--no-deps` install doesn't help either, because `impresso_essentials.io.s3` does `import dask.bag as db` at module level. Full rationale in `.progress/io-layer/notes.md` and step 12 of `.progress/plan.md`.

Instead, we **vendor** the three small helpers we actually used:

- `get_s3_client`, `get_s3_resource`, `upload_to_s3` → reimplemented in `src/impresso_text_embedder/io.py` (~40 lines, thin boto3 wrappers reading `SE_ACCESS_KEY`/`SE_SECRET_KEY`/`SE_HOST_URL` from env).

Two streaming/failure-mode choices we kept even before vendoring (see `.progress/io-layer/notes.md`):

- `read_jsonlines` (upstream) is **not** used — it does `body.read()` + `bz2.decompress(data)` and loads the full file into memory, which breaks the streaming/prefetch model. Our `io.iter_jsonl_bz2` opens the S3 body stream and wraps it with `bz2.open` for chunked decoding.
- The upstream `upload_to_s3` returns `bool` and swallows exceptions; our `io.upload_local_file` raises on failure so a botched upload can't masquerade as a persisted output.

If we ever need something else from `impresso-essentials` (e.g. the `versioning` manifest), prefer adding a **minimal vendored snippet** or making the upstream release a runtime-slim extras set; do not re-introduce the full dep against the NGC image.

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
│       ├── io.py               # boto3 S3 helpers (vendored from impresso-essentials) + streaming reader
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

## Commands

```bash
uv sync --extra dev
uv run impresso-embed-create --provider SNL --input-bucket <in> --output-bucket <out> --embedding-level text --batch-size 64
# Local dev: default log path is /rcp-scratch/... (PVC). Override with --log-dir:
uv run impresso-embed-create --provider SNL --input-bucket <in> --output-bucket <out> --log-dir ./logs
uv run impresso-embed-validate s3://.../EXP-1912.jsonl.bz2                                   # structural
uv run impresso-embed-validate s3://.../EXP-1912.jsonl.bz2 --target s3://.../golden/...      # comparison (tol=1e-4)
uv run pytest
uv run ruff check .
```

---

## Containerised execution (EPFL RCP / Run:AI)

Production runs happen in a container submitted to Run:AI on EPFL RCP. The
`Dockerfile` builds on `nvcr.io/nvidia/pytorch:25.03-py3` (NGC torch is
left untouched — our package's `torch>=2.2` is already satisfied), creates
an LDAP-matched user (so PVC writes land with the right ownership), and
sets `ENTRYPOINT ["impresso-embed-create"]`. No bash wrapper.

A slim `Makefile` orchestrates the container/cluster side only:

```bash
cp .env.docker.example .env.docker      # fill LDAP UID/GID, registry, RUNAI_PROJECT
make docker-login                       # once
make docker-build-push                  # build linux/amd64 + push to Harbor
make k8s-create-secrets                 # S3 creds + Harbor pull secret (idempotent)
make runai-submit PROVIDER=BNL \
     INPUT_BUCKET=22-rebuilt-final \
     OUTPUT_BUCKET=42-processed-data-final \
     EMBED_EXTRA_ARGS="--embedding-level text --batch-size 64"
make runai-interactive && make runai-bash    # debug pod (sleep infinity entrypoint)
make runai-delete-debug                       # cleanup
```

Design rationale, gotchas, and the secret-name conventions live in
`.progress/docker-runai/notes.md`. The Makefile contains **no data
processing logic** — it is only `docker buildx` / `kubectl` / `runai`
glue, with `make help` listing every target.

---

## Non-goals (explicit)

- Multi-GPU, multi-node, DDP.
- Non-Ampere/Hopper GPU variants (V100, Blackwell) — deferred.
- Incremental/partial-file recovery mid-shard. A file either completes or is redone.
- Local-only workflows beyond tests — production flow is S3 in, S3 out.
- A new Make-driven **data pipeline**. The `main`-branch stamp/sync orchestration is being replaced by the Python CLI. (A small `Makefile` exists but it only wraps `docker buildx`, `kubectl`, and `runai submit` — no data flows through it. See "Containerised execution" below.)

---

## Decisions recorded

- **Validation metric:** cosine distance on L2-normalized vectors; default tolerance `1e-4`. Rationale in `.progress/validation-metric/notes.md`.
- **Model slug:** Impresso versioned-embedding convention — `Alibaba-NLP/gte-multilingual-base → embeddings_gte_v1-1-0` via `pipeline.MODEL_SLUG_OVERRIDES`. Unknown models fall back to stripping the HF vendor prefix. Revision is **not** appended to the slug; different revisions of the same model go to the same output path. When a new Impresso-blessed model is added, register its slug in the override map.
- **JSON codec:** `orjson` on both read and write paths (`orjson.loads` on input lines, `orjson.dumps(..., option=OPT_APPEND_NEWLINE)` on output). Output bz2 file is opened in binary mode (`bz2.open(path, "wb")`) since orjson returns bytes. Embeddings rounded to 5 decimals on write (handled by `schema.to_dict`). Rationale in `.progress/io-throughput/notes.md`.
- **Per-provider pipeline overlap:** `process_provider` drives two single-slot `ThreadPoolExecutor`s — one prefetcher downloading file N+1 while file N encodes, and one uploader sending file N's output while file N+1 encodes. Backpressure is implicit (wait on the previous upload future before submitting the next). Inputs are fetched with `boto3.s3.transfer.TransferConfig(multipart_threshold=8 MB, multipart_chunksize=8 MB, max_concurrency=10, use_threads=True)`; this hits Ceph RadosGW (Switch Engines) with parallel ranged GETs. Rationale in `.progress/io-throughput/notes.md`.
- **Per-file telemetry:** `telemetry.StageTimer` + `telemetry.GpuSampler` emit one INFO line per completed file with `download_s`, `encode_s`, `upload_wait_s` and `gpu_util_mean`/`p10`. `GpuSampler` soft-imports `pynvml` (bundled by NGC) and no-ops cleanly when it's missing. Rationale in `.progress/io-throughput/notes.md`.
- **Streaming inputs:** hand-rolled (see `.progress/io-layer/notes.md`); `impresso_essentials.io.s3.read_jsonlines` is intentionally unused on the hot path.
- **A100 bf16 strategy:** autocast around `encode()`, model weights stay fp32. Recorded in `.progress/gpu-throughput/notes.md`. Revisited in `.progress/gpu-profiles/notes.md` (Q2): stays the recipe on H100 too — xformers dispatches on Q/K/V dtype, which is bf16 inside the autocast region regardless of weight dtype.
- **GPU profile detection:** `accel.py` reads `torch.cuda.get_device_capability()` at model-load time and returns a profile with the default batch size. cc `(8,0)`→A100=64, cc `(9,0)`→H100/H200=128, others→32 with a warning, CPU→8. Runtime detection (not build-time) so one image runs on either arch and the startup log records what landed. Rationale in `.progress/gpu-profiles/notes.md`.
- **xformers + unpadding wired as the default fast path:** when xformers is importable and the device is CUDA, we pass `config_kwargs={"unpad_inputs": True, "use_memory_efficient_attention": True}` to `SentenceTransformer` — not `model_kwargs`. Alibaba's custom modeling file reads these off `self.config`, and sentence-transformers v5 pre-loads the config and passes it explicitly to `from_pretrained`, which makes HF skip the kwarg-to-config routing that `model_kwargs` would have relied on; forwarding through `config_kwargs` lands them on the config object via `AutoConfig.from_pretrained(..., **config_kwargs)`. The modeling file then routes through `xformers.ops.memory_efficient_attention`. HF-generic `attn_implementation` does **not** work for this model (the custom modeling file rejects it with `NewModel does not support Flash Attention 2.0 yet`). NGC 25.03 does not bundle xformers, so the Dockerfile pins `xformers==0.0.30` from PyTorch's cu128 wheel index with `--no-deps`. xformers dispatches to FA2/FA3 only when `flash-attn` is also installed; today we install xformers alone (CUTLASS kernel — still a real speedup with unpadding), with `flash-attn` deferred as a measurable lever. Full research in `.progress/gpu-profiles/notes.md`.
- **Re-embed on input change:** compare S3 `LastModified` of input vs existing output at skip-decision time. Input timestamps come from `list_objects_v2` (no extra HEAD), output via `head_last_modified`. `--force` overrides unconditionally. Known gap: byte-identical re-uploads still trigger a re-embed, and this does not replace the deferred `impresso-essentials.versioning` manifest. Rationale in `.progress/reembed-on-change/notes.md`.
- **impresso-essentials vendored, not imported:** the package is **not** in `pyproject.toml` and not installed in the Docker image. Its 1.4.x metadata hard-pins `numpy==2.2.1` (plus dask, pandas, pyarrow…), which would uninstall NGC's `numpy==1.26.4` and break the apex/NCCL/transformer-engine/xformers ABI stack. A `--no-deps` install also fails because `impresso_essentials.io.s3` does `import dask.bag as db` at module level. The three helpers we used (`get_s3_client`, `get_s3_resource`, `upload_to_s3`) are vendored into `src/impresso_text_embedder/io.py` — ~40 lines of boto3 wrappers reading `SE_*` env vars. Full rationale in step 12 of `.progress/plan.md` and `.progress/io-layer/notes.md`.
- **Text-level output schema aligned with Impresso document-embeddings schema:** required `{ci_id, model_id, embedding, size}`, optional `{ts, ci_type}`. Previously wrote `{id, ts, embedder, len, embedding, text?}` — all four divergences fixed (renames + drops + added `size`). `--include-text` CLI flag removed with the field. The `model_id` *value* still carries the `name@revision` tag from `build_embedder_tag`; aligning that string with the Impresso slides convention (e.g. `doc-embeddings_<slug>-vX-Y-Z`) is a deferred follow-up. Rationale in `.progress/create-cli/notes.md`.
- **Upload integrity:** disable boto3's default flex-checksums with `Config(request_checksum_calculation="when_required", response_checksum_validation="when_required")` — the `when_supported` default makes `PutObject` send `aws-chunked` with no `Content-Length`, which Ceph RadosGW (Switch Engines) rejects with `MissingContentLength`. Needs `boto3>=1.36.5` (pin bumped) so the setting propagates through the TransferManager used by `upload_file`. Each upload is then verified by streaming a local MD5, HEAD-ing the uploaded object, and checking `ContentLength` against local size plus (single-part only) ETag against MD5; multipart ETag reconstruction is deferred. On mismatch the bad object is best-effort-deleted so `--skip-if-s3-exists` doesn't hide the failure. Rationale in `.progress/upload-integrity/notes.md`.
- **Transformers pinned `<5`:** `transformers>=5` corrupts `position_ids` on `Alibaba-NLP/new-impl` (the `trust_remote_code` modeling repo behind `gte-multilingual-base`) because v5's meta-device loading skips re-initialising `persistent=False` buffers, and Alibaba registers `position_ids` that way. The crash surfaces as a CUDA IndexKernel assert at `modeling.py:392 rope_cos[position_ids]` on **any** input — including the Makefile's 6-token smoke test — so it is not a sequence-length bug. Pinned `transformers>=4.46,<5` and `sentence-transformers>=3.0,<5.2` in `pyproject.toml`, re-asserted in the Dockerfile as `transformers>=4.46,<5` + `sentence-transformers>=5.0,<5.2` + a build-time `assert transformers.__version__.startswith('4.')` guardrail. Upstream trackers: HF transformers #43950, #44534; model discussion #30. `Alibaba-NLP/new-impl` is unmaintained (last commit Aug 2024), so the fix lives on our side until we either see an upstream v5 fix or switch embedder. Full rationale — including the red herrings we ruled out (sequence length, xformers unpad path, BL-specific content, NGC image mismatch) — in `.progress/transformers-v5-regression/notes.md`.
- **Model revision pinned to `f7d567e`:** `Alibaba-NLP/gte-multilingual-base` is loaded at a specific HF commit so upstream re-tags can't silently change the weights and invalidate the `--tol 1e-4` cosine validation contract. Single source of truth: `DEFAULT_MODEL_REVISION` in `src/impresso_text_embedder/model.py`; `load_model` and the `--model-revision` CLI default both read it. The `Makefile` mirrors the pin as `CREATOR_NAME`/`HF_MODEL_NAME`/`HF_MODEL_VERSION` (`?=` so `.env.docker` overrides win) and `runai-submit` forwards them as explicit `--model-name` + `--model-revision`, which makes the pin recoverable from `runai describe job` even if the image changes. Output slug stays revision-agnostic (`embeddings_gte_v1-1-0`) — the pin is about reproducibility, not about forking output paths. Rationale, bump workflow (one-off in `.env.docker` vs permanent in `model.py`), and deferred `huggingface_hub` SHA-verify follow-up in `.progress/model-revision-pin/notes.md`.
- **Structured logging split file ↔ terminal:** `impresso-embed-create` writes a full INFO log to `/rcp-scratch/<username>/experiments/embeddings/<YYYY-MM-DD>/<provider>.log` (override with `--log-dir <path>`; file level via `--log-level-file`, default `INFO`). The terminal shows a single `tqdm` bar over the planned files with per-file `set_postfix(dl=…s enc=…s up=…s gpu=…%)` from `StageTimer`/`GpuSampler`; only `ERROR` records reach the terminal, via a `TqdmLoggingHandler` that calls `tqdm.write` so stack traces don't shred the bar. If `/rcp-scratch` is not mounted and no `--log-dir` is given, the CLI exits non-zero with a clear "PVC not mounted" message — no silent fallback (dev fallback to `./logs/` was rejected because it would mask a PVC misconfig on RCP). Username via `getpass.getuser()`, which matches `LDAP_USERNAME` inside the container. Scope: `impresso-embed-create` only; `impresso-embed-validate` is a one-shot diagnostic and keeps its old stderr logging. Full rationale, rejected alternatives (per-run timestamped dirs, structured JSON logs, rotating file handler), and the tqdm-postfix format choice in `.progress/structured-logging/notes.md`.
- **Record filtering — `missing_content_type` distinct from `content_type`:** records with `tp not in cfg.content_types` and records with `tp` absent are tallied under separate reasons. Missing `tp` is anomalous enough to deserve a one-shot WARNING per file (first occurrence only; later ones DEBUG) so data-quality drift surfaces without spamming the log. The existing `telemetry.format_stats_line` surfaces both counts in the per-file `done` line (`skipped=N (content_type=X missing_content_type=Y …)`). Restores parity with legacy `a433970:lib/text_embedding_processor.py:257-259`, which dropped `None not in ["ar"]` → True under `skipped_type_None`; the migration's `_passes_content_type` had loosened this to pass missing `tp` through. Rationale, drift-from-legacy table for every filter reason, and rejected alternatives (per-record WARN, summary-only WARN, CLI escape hatch) in `.progress/record-filtering/notes.md`.
- **Long-doc handling at `--embedding-level text` — partial landing, extensible by design:** architecture is two orthogonal registries — `impresso_text_embedder.chunking` (token-aware chunkers) and `impresso_text_embedder.aggregation` (chunk-vector → doc-vector). Both registries accept kwargs so future strategies inject runtime deps (tokenizer, decay coefficient, …) without CLI-surface churn. Today the CLI wires `chunking="fixed-window"` (dumbest-thing-that-works: tokenize → slice into contiguous non-overlapping windows of `tokenizer.model_max_length - num_special_tokens_to_add(pair=False)` ids → decode; no sentence regex, no overlap, no chonkie) + `aggregation="mean"` (mean pool + L2 renorm). `chunking="token-budget"` (sentence-aware greedy packer) is also registered but opt-in — reachable via `get_strategy("token-budget", …)` for A/B testing once `fixed-window` has a baseline. Additional strategies land as a new module + one `register_strategy` line + one `choices=` tuple entry. Wiring: `EncoderConfig.long_doc: LongDocConfig | None` (default `None` = legacy behaviour); when active, `TextBatcher` detects long docs via a cheap chars-per-token pre-filter then a real tokenise, chunks via the registered chunker, rides the K chunk texts on the same batched `encode_texts` call as short docs, and aggregates K→1 vectors per-record on flush. CLI flags: `--long-doc-strategy {truncate,chunk}` (default **`chunk`** — long docs no longer silently truncated; `truncate` is the opt-out for reproducing pre-step-16 outputs), `--long-doc-chunk-tokens N` (default **`None`** — auto-derived at CLI-init time from the loaded model's tokenizer as `model_max_length - num_special_tokens_to_add(pair=False)`; 8190 for gte-multilingual-base; `_FALLBACK_CHUNK_TOKENS=8000` used only if the tokenizer doesn't advertise the attrs), `--long-doc-aggregation {mean}` (only choice today). **Boundary decision**: chunking only fires when `tokens(doc) > model_max_tokens`; for docs ≤8192 the one-shot path wins — `gte-multilingual-base` uses CLS pooling (confirmed via `1_Pooling/config.json`) and is natively trained at 8192 with RoPE, so chunk-and-pool strictly drops cross-chunk attention without a mechanism to recover it, and Jina's own "No Chunking" benchmark beats both naive and late chunking on NFCorpus. **Two independent L2 normalizations, not one.** The encoder applies per-chunk L2 normalization via the model's built-in `2_Normalize` module (or via `--normalize-embeddings=True`, a no-op on this model). MeanPoolStrategy applies a second L2 renormalization on the *aggregated* mean vector — because the mean of K unit vectors is not itself unit-norm. Different inputs, different necessity, different pipeline stages; do not fold one into the other. Full analysis in the "Two independent L2 normalizations" section of `.progress/long-doc-chunking/notes.md`. Late Chunking (Jina 2024) **rejected** for this embedder: it assumes a mean-pooled model, and applying it to a CLS-pooled one puts the output in a subspace the training loss didn't shape; reconsider only if we switch embedders. Full design space (11 chunking options A–K, 8 aggregation options α–θ, literature anchors, CLS-pooling deep-dive, shipped/deferred split) in `.progress/long-doc-chunking/notes.md`.

## Still open — needs real GPU time

- **Default `--batch-size` per profile and per embedding level.** Current placeholders in `accel.py`: A100=64, H100/H200=128. Effective range on H100 is probably 128–256+. Measure once, update the profile defaults, document here.
- **Confirm xformers' FA3 kernel actually fires on H100.** Detectable via a one-shot probe at model-load (call `xformers.ops.memory_efficient_attention` on a dummy tensor and inspect the dispatched op). If it silently falls back to the CUTLASS kernel we're leaving the big H100 win on the table.
- **Numerical-drift validation A100↔H100** at `--tol 1e-4` cosine. Generate a golden on A100, re-embed the same input on H100, validate. Record the outcome in `.progress/gpu-profiles/notes.md`.
- **RCP node-type labels** for `RUNAI_GPU_TYPE=…`. The Makefile plumbing is in; the exact label values for H100 vs A100 on EPFL RCP need to be confirmed and recorded in `.progress/gpu-profiles/notes.md`.
- **Data manifest (impresso-essentials `versioning`) emission alongside outputs.** Useful but not required for any current consumer — add when one needs it.
