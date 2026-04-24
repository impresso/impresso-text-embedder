# Migration plan — impresso-text-embedder

Living step list. Statuses: `todo` / `wip` / `done` / `deferred`. Slug names are stable; order can shift. See `/Users/adrien/.claude/plans/graceful-wishing-lake.md` for the mechanism this follows, and `CLAUDE.md` at repo root for package intent.

| # | Slug | Status | Notes folder? |
|---|---|---|---|
| 1 | package-skeleton | done | no |
| 2 | io-layer | done | `.progress/io-layer/` |
| 3 | schema-text-rebuild | done | no |
| 4 | model-encoder | done | `.progress/gpu-throughput/` |
| 5 | chunking | done | `.progress/chunking/` |
| 6 | create-cli | done | `.progress/create-cli/` |
| 7 | validate-cli | done | `.progress/validation-metric/` |
| 8 | e2e-docs | done | no |
| 9 | docker-runai | done | `.progress/docker-runai/` |
| 10 | reembed-on-change | done | `.progress/reembed-on-change/` |
| 11 | gpu-profiles | done | `.progress/gpu-profiles/` |
| 12 | drop-impresso-essentials | done | `.progress/io-layer/` |
| 13 | io-throughput | done | `.progress/io-throughput/` |
| 14 | model-revision-pin | done | `.progress/model-revision-pin/` |
| 15 | structured-logging | done | `.progress/structured-logging/` |
| 16 | long-doc-chunking | partial | `.progress/long-doc-chunking/` |

## Step details

### 1. package-skeleton
`pyproject.toml` (hatchling + uv), `src/impresso_text_embedder/` layout, Python ≥3.10, ruff config, `LICENSE` (AGPL-3.0-or-later reused from `main`), `tests/` with placeholder, short `README.md` linking to `CLAUDE.md`, `.gitignore` update, delete `.flake8`. Verify: `uv sync` resolves, `uv run pytest` green.

### 2. io-layer
Wrap `impresso_essentials.io.s3` for `.jsonl.bz2` streaming, provider/alias/year iteration, idempotent skip-if-exists. Dotenv loaded at CLI boundary only.

### 3. schema-text-rebuild
Port `rebuild_ft_from_offsets`, `rebuild_sentence_from_offsets` verbatim. Dataclass/pydantic types for the three output schemas from `main:lib/text_embedding_processor.py`.

### 4. model-encoder
SentenceTransformer load + A100 optimizations (bf16 autocast, optional flash-attn, tunable batch size). Measurement deferred to real-GPU session; unit tests assert API shape only. Populate `.progress/gpu-throughput/notes.md`.

### 5. chunking
Registry + chonkie semantic strategy (threshold 0.5, chunk_size 1024, min_sentences 5). Contract in `.progress/chunking/notes.md`.

### 6. create-cli
`impresso-embed-create --provider ...` orchestrator, 1:1 mapping, async prefetch + upload overlap. Flags listed in `CLAUDE.md`.

### 7. validate-cli
`impresso-embed-validate <path> [--target ...] [--tol ...]`. Metric + default tol decided and documented in `.progress/validation-metric/notes.md`.

### 8. e2e-docs
Tiny local fixture end-to-end; polish `README.md`; resolve or explicitly defer remaining "Things to decide" items in `CLAUDE.md`.

### 9. docker-runai
Container image + Run:AI submission for EPFL RCP. Reference: `feat/docker` branch. Deliverables:
- `Dockerfile` based on `nvcr.io/nvidia/pytorch:25.03-py3`, LDAP-matched user (PVC ownership), `pip install .` from copied source, `ENTRYPOINT ["impresso-embed-create"]`.
- `.env.docker.example` (LDAP UID/GID, registry/project, Harbor robot creds).
- `Makefile` (slim — docker build/push, k8s secrets, runai submit + interactive debug). No data-processing logic; the CLI handles that.
- `.gitignore` entries for `.env.docker` and `config.local.mk`.
- `CLAUDE.md` section linking to the workflow.
Decisions and gotchas in `.progress/docker-runai/notes.md`.

### 10. reembed-on-change
Re-embed an output when its input has been re-uploaded on S3, without resurrecting the old stamp tree or taking on the full `impresso_essentials.versioning` manifest system yet. Mechanism: compare S3 `LastModified` of input vs existing output at skip-decision time. `--force` still overrides unconditionally.
- `io.py`: add `last_modified: datetime | None` to `InputKey`, populate from `list_objects_v2` (no extra HEAD). Add `head_last_modified(bucket, key) -> datetime | None` (None on 404).
- `pipeline.py`: rewrite the skip branch in `process_file` and its dry-run twin in `process_provider` to compare timestamps. Distinct log lines for "skip (up-to-date)" vs "reprocess (input newer)".
- Tests: cover {output missing, output newer, output older, --force overrides}. Update existing `object_exists` patches to the new helper where the skip path is exercised.
- `CLAUDE.md`: record the decision under "Decisions recorded"; note the known gap (versioning manifest still deferred).
Rationale, trade-offs, and the explicit choice against option 2/3/4 in `.progress/reembed-on-change/notes.md`.

### 11. gpu-profiles
Support A100 and H100 (incl H200) from the same codebase and the same container image, picking the right default batch size per device. Attention kernel selection is transparent: xformers' `memory_efficient_attention` dispatches to FA3 on Hopper and FA2 on Ampere on its own, based on Q/K/V dtype inside the existing bf16 autocast region. HF-generic `attn_implementation` is **rejected** by the model (`NewModel does not support Flash Attention 2.0 yet`) — the Alibaba-NLP fast path requires `unpad_inputs=True` + `use_memory_efficient_attention=True` set on the model config via `config_kwargs` (not `model_kwargs` — ST v5 pre-loads the config and passes it explicitly, which skips HF's kwarg-to-config routing). Research and feature-comparison table in `.progress/gpu-profiles/notes.md`.
- `src/impresso_text_embedder/accel.py`: new module. `Profile` dataclass (`name`, `default_batch_size`, `notes`). `detect_profile()` reads `torch.cuda.get_device_capability()`: `(8,0)`→A100 batch 64, `(9,0)`→H100/H200 batch 128, other-CUDA→batch 32 with a warning, no-CUDA→batch 8. `has_xformers()` probes the import. `log_profile()` emits one INFO line at model load. All cached with `lru_cache`.
- `model.py`: call `log_profile()` at load time. When CUDA + xformers importable, pass `config_kwargs={"unpad_inputs": True, "use_memory_efficient_attention": True}` to `SentenceTransformer`; omit otherwise (keeps CPU / unit-test path unchanged). Must be `config_kwargs` rather than `model_kwargs` so the flags land on `self.config` before `NewModel.__init__` runs — see note for why the `model_kwargs` path is broken under ST v5. Existing fp32-weights + bf16-autocast strategy stays — xformers dispatches on Q/K/V dtype and Q/K/V are bf16 inside the autocast region.
- `cli/create.py`: `--batch-size` default → `None`; resolved in `main()` from `detect_profile().default_batch_size`. Explicit flag still overrides.
- `pyproject.toml`: add `accel = ["xformers>=0.0.28"]`. Single extra, not two. Matters for non-NGC local dev.
- `Dockerfile`: NGC `pytorch:25.03-py3` does **not** bundle xformers (verified against the release notes' component list — an earlier note in this plan wrongly claimed it did). Install it explicitly: `pip install --no-deps "xformers==0.0.30" --index-url https://download.pytorch.org/whl/cu128`. `--no-deps` avoids pip re-installing torch. Followed by a build-time `python -c "import xformers"` assertion (mirrors the numpy guard). xformers alone gives a CUTLASS memory-efficient kernel + unpadding; the FA2/FA3 dispatch requires the separate `flash-attn` package — deferred as a measurable lever. One image, not two — it's arch-agnostic; the encoder auto-detects at runtime.
- `Makefile`: add `RUNAI_GPU_TYPE ?=` and thread `--node-type $(RUNAI_GPU_TYPE)` onto `runai submit` / `runai-interactive` when set. Documented in `make help`. Exact RCP node-type labels TBD (OPEN in notes).
- Tests: `tests/test_accel.py` unit-tests profile selection with `unittest.mock.patch` on `torch.cuda.is_available` / `get_device_capability`, plus `has_xformers` toggling via `sys.modules`. No live-GPU assertions.
- Validation (deferred to live-GPU time): regenerate a golden on A100, re-embed the same input on H100, confirm `impresso-embed-validate` passes at `--tol 1e-4` cosine. Also probe whether FA3 actually fires on H100. Record the outcome in `.progress/gpu-profiles/notes.md`.
- `CLAUDE.md`: "Target hardware" reframed as A100+H100; decisions for profile detection + xformers+unpadding path wired; FP8/TE + flash-attn-standalone listed as deferred levers.
Rationale (runtime-detection-not-build-time, why xformers is the only fast path, A100→H100 feature table, ~2.0–2.8× expected end-to-end speedup) in `.progress/gpu-profiles/notes.md`.

### 12. drop-impresso-essentials
Drop the `impresso-essentials` dependency entirely. Blocker: building on `nvcr.io/nvidia/pytorch:25.03-py3`, `pip install .` fails because `impresso-essentials==1.4.1` hard-pins `numpy==2.2.1` in its metadata, which would uninstall NGC's `numpy==1.26.4` and break the apex/NCCL/transformer-engine/xformers stack (all compiled against numpy-1.x ABI). A `--no-deps` install doesn't work either: `impresso_essentials/io/s3.py` does `import dask.bag as db` at module level, and dask isn't a dep we can satisfy without reintroducing numpy 2.x (and pandas, pyarrow…). The three symbols we consume (`get_s3_client`, `get_s3_resource`, `upload_to_s3`) don't use dask at all — they're ~40 lines of boto3 wrappers around `SE_*` env vars. Vendor them.
- `src/impresso_text_embedder/io.py`: replace `from impresso_essentials.io.s3 import …` with local definitions of `get_s3_client`, `get_s3_resource`, `upload_to_s3`. Read `SE_ACCESS_KEY`/`SE_SECRET_KEY`/`SE_HOST_URL` from env (dotenv is already loaded once at the CLI boundary — don't re-call `load_dotenv()`). Keep the `upload_local_file` wrapper that raises on `False` return.
- `pyproject.toml`: `impresso-essentials>=1.4.1` already removed. `boto3>=1.34`, `smart-open[s3]>=7.0`, `python-dotenv>=1.0` cover the runtime needs.
- `Dockerfile`: remove the `pip install --no-deps "impresso-essentials==1.4.1"` line and the `from impresso_essentials.io.s3 import …` smoke test. Keep the `numpy.__version__.startswith('1.26')` assertion as a guardrail against silent upgrades.
- `CLAUDE.md`: drop the `uv pip install --no-deps …` step from Commands; rewrite the "Reuse `impresso-essentials`" section to explain that the three S3 helpers are vendored (cite this step); update the "Decisions recorded" bullet so it reflects vendoring, not `--no-deps`.
- `.progress/io-layer/notes.md`: update the "What we reuse" table (now: no functions reused — three are vendored) and add a subsection "Why we dropped impresso-essentials" with the numpy-ABI + dask-top-level-import argument.
- Tests: `tests/test_io.py` already patches functions as attributes on our own `io` module, not on `impresso_essentials`. Verify it still passes unchanged.
- Verify: `uv pip uninstall impresso-essentials`, `uv run pytest`, `uv run impresso-embed-create --help`. Then confirm the image with `make docker-build`.
Rationale in `.progress/io-layer/notes.md`.

### 13. io-throughput
Close the IO/CPU half of the "GPU must be the bottleneck" target from
`CLAUDE.md` — `.progress/create-cli/notes.md` explicitly deferred
prefetch/upload overlap here. Scope is CPU- and IO-side only; GPU-side
decisions live in `.progress/gpu-throughput/` and `.progress/gpu-profiles/`.

**Shipped (Tier A)**, full catalogue of alternatives and rationale in
`.progress/io-throughput/notes.md`:
- `telemetry.py`: `StageTimer` context manager + `GpuSampler` background
  thread (2 Hz, `pynvml` soft-imported, NGC ships it) + `format_stats_line`.
  Emits one INFO line per completed file with `download_s`, `encode_s`,
  `upload_wait_s`, `gpu_util_mean`/`p10`/`n`.
- `pipeline.process_provider`: rewritten around two
  `ThreadPoolExecutor(max_workers=1)` — prefetch and upload. File N+1
  downloads while file N encodes; file N uploads while file N+1 encodes.
  Skip logic pre-applied before prefetch via `_plan_files`.
- `pipeline.py` + `validate.py`: `json` → `orjson` on read and write.
  Output bz2 opened in binary mode (orjson returns bytes); `OPT_APPEND_NEWLINE`
  replaces the manual `\n` write. One test behaviour change: orjson
  rejects bare `NaN` at parse time — `test_catches_nan` now asserts
  non-empty errors rather than a specific "non-finite" message.
- `io.py`: `download_to_local` (multipart ranged GETs), `iter_jsonl_bz2_path`
  (local stream), `DEFAULT_TRANSFER_CONFIG` (8 MB / 8 MB / 10 threads).
- Regression guard: `tests/test_pipeline.py::test_process_provider_overlaps_prefetch_and_upload`
  uses two `threading.Event`s to pin file N's upload mid-flight and assert
  file N+1's download has already started.
- `CLAUDE.md` "Decisions recorded" updated with JSON codec, pipeline
  overlap + `TransferConfig` defaults, and per-file telemetry.

**Not implemented — potential follow-ups** (see Tier B / Tier C / Known
deferred from Tier A sections of `.progress/io-throughput/notes.md`):
`indexed_bzip2` parallel decompression, in-file reader thread, output
codec swap (bz2→zstd, cross-team), token-budget batching, `msgspec`,
pre-tokenisation on CPU threads, chunker pre-compute thread,
`OPT_SERIALIZE_NUMPY` on the output side, `--transfer-concurrency` /
`--multipart-chunksize` CLI flags, CLI escape hatch to disable the
prefetch/upload overlap.

Verify once observed on real hardware: GPU SM utilization ≥85% during
steady-state encode on A100 (matches the acceptance bar in `CLAUDE.md`);
no regression in `impresso-embed-validate` at `--tol 1e-4`. Both are
acceptance checks, not gating for code; the code landed with 151/151
tests green and ruff clean.

### 14. model-revision-pin
Pin the HF revision of `Alibaba-NLP/gte-multilingual-base` to `f7d567e`
so runs are reproducible and the `--tol 1e-4` cosine validation contract
is stable against upstream re-tags. Single source of truth lives in
`src/impresso_text_embedder/model.py` as `DEFAULT_MODEL_REVISION`;
`load_model(revision=DEFAULT_MODEL_REVISION, …)` and
`cli/create.py` (`--model-revision` default) both read it. The
`Makefile` mirrors the pin as `CREATOR_NAME`/`HF_MODEL_NAME`/
`HF_MODEL_VERSION` (`?=` so `.env.docker` wins) and `runai-submit`
forwards them as explicit `--model-name` + `--model-revision`, which
makes the pin recoverable from `runai describe job` even if the image
changes. Tests: three assertion strings updated for the new default
(`tests/test_cli_create.py::test_main_non_dry_run_loads_model`,
`tests/test_e2e.py`, and a new
`test_parser_pins_model_revision_by_default` that guards against
accidental un-pin); `tests/test_pipeline.py::_cfg` still passes
`model_revision=None` so its `@default` assertions are unaffected by
the CLI default. Output slug (`embeddings_gte_v1-1-0`) intentionally
stays revision-agnostic — matches the existing "Decisions recorded"
note in `CLAUDE.md`. Rationale, bump workflow, and known gaps
(`huggingface_hub` SHA-verify is a cheap deferred follow-up) in
`.progress/model-revision-pin/notes.md`.

### 15. structured-logging
Split runtime logs into a file (full detail) and the terminal (a `tqdm`
progress bar over the planned files + ERROR records only), so a Run:AI
job writes a searchable log to scratch while the submitting user sees a
clean progress readout.

**Log file path.** Default
`/rcp-scratch/<username>/experiments/embeddings/<YYYY-MM-DD>/<provider>.log`.
Username comes from `getpass.getuser()` — inside the container this
matches `LDAP_USERNAME` because the Dockerfile creates and runs as that
user. `--log-dir <dir>` overrides the base path (keeps the
`<YYYY-MM-DD>/<provider>.log` suffix). If `/rcp-scratch/` does not exist
and no `--log-dir` override is given, the CLI exits non-zero with a clear
error ("PVC not mounted — pass --log-dir or mount /rcp-scratch"). No
silent fallback to `./logs/` or similar; test runs use `--log-dir` or
`tmp_path` fixtures.

**Terminal policy.** One `tqdm.tqdm` bar over the post-skip
`to_process` list, `file=sys.stderr`. `pbar.set_description(alias/year)`
while a file is encoding; after each file completes,
`pbar.set_postfix(dl=…s enc=…s up=…s gpu=…%)` with the `StageTimer`
totals so the user sees live throughput. ERROR-level records reach the
terminal through a custom `TqdmLoggingHandler` that calls `tqdm.write()`
(one line above the bar, doesn't shred it). No INFO/WARNING/DEBUG on
the terminal. Bar auto-disables on non-tty (`disable=None`), so tests
and pipes don't emit progress noise.

- `src/impresso_text_embedder/logging_setup.py`: new module.
  `configure_logging(provider, log_dir=None, log_level_file="INFO") -> Path`.
  Resolves the log path, creates the date directory, attaches a
  `FileHandler` at the requested level and a `TqdmLoggingHandler` at
  ERROR. Called once from `cli/create.py`'s `main()`. Returns the path so
  `main()` can print `"logging to <path>"` to stderr (plain `print`, not
  logging) on the first line before the bar starts.
- `src/impresso_text_embedder/cli/create.py`: replace `--log-level` with
  `--log-level-file` (default `INFO`) and add `--log-dir <path>`. Drop the
  existing `logging.basicConfig(...)` call; `configure_logging` owns the
  root logger now. Print the resolved log path to stderr on startup and
  the processed/skipped summary to stderr at exit (both plain `print`, so
  they show regardless of handler levels).
- `src/impresso_text_embedder/pipeline.py`: wrap the `process_provider`
  file loop with `tqdm(total=len(to_process), file=sys.stderr, disable=None)`.
  After each file completes, pull `timer.totals` + `gpu.summary()` and
  `pbar.set_postfix(...)`. The existing `format_stats_line` INFO line
  stays — it goes to the file handler, not the terminal. Do **not**
  import `tqdm` at module top-level gated behind a test (keep it as a
  normal dependency; the bar is always opt-out via non-tty detection).
- `pyproject.toml`: add `tqdm>=4.66` to runtime deps.
- Tests:
  - `tests/test_logging_setup.py`: (a) no `/rcp-scratch/` + no override
    → `SystemExit` with message mentioning `--log-dir`; (b) `--log-dir
    tmp_path` creates the file at `<tmp>/<date>/<provider>.log` and INFO
    records land in it; (c) ERROR records reach the `TqdmLoggingHandler`
    (patch `tqdm.write` and assert call); (d) INFO records do **not**
    reach the terminal handler.
  - `tests/test_cli_create.py`: assert the parser has `--log-dir` and
    `--log-level-file`; delete / update the reference to `--log-level`.
  - `tests/test_pipeline.py`: keep existing behaviour (tqdm
    auto-disables on non-tty stderr in CI). Add one assertion that
    `pbar.set_postfix` is called per completed file by patching
    `impresso_text_embedder.pipeline.tqdm`. The existing
    prefetch/upload-overlap regression test keeps the same threading
    gates.
- `CLAUDE.md`: add a "Decisions recorded" entry summarising the file
  layout, fail-fast policy, and terminal-ERROR-only policy. One-line
  addition under "Commands" showing `--log-dir ./logs` for local dev.
- `Makefile`: no change needed — the container already has
  `/rcp-scratch` mounted via PVC, so the default path resolves cleanly.
  (A follow-up may surface `--log-dir` as an extra knob, but the default
  is already the right shape.)

Rationale and the rejected alternatives (per-run timestamped dir, dev
fallback to `./logs`, structured JSON logs) in
`.progress/structured-logging/notes.md`.

### 16. long-doc-chunking
At `--embedding-level text`, documents longer than the model's 8192-token
max context are silently truncated by the HF tokenizer's default
`truncation=True` (see `embed.py` → `TextBatcher.flush` → `encode_texts`).
No warning, no tally, no trace in the output. For the Impresso long tail
(feuilletons, parliamentary records, full-page speeches) this means the
embedding only represents the head of the document.

**Partial landing (2026-04-23).** The architecture is in place and
exercised end-to-end for one concrete combination:
(`chunking=fixed-window`, `aggregation=mean`). Status is "partial"
on purpose — adding more strategies is a small incremental step and
the framework is designed to make that cheap. Full shipped/deferred
list in the "Implementation status" section at the top of
`.progress/long-doc-chunking/notes.md`. Summary:

- **Shipped:**
  - `aggregation/` module with a kwargs-capable registry mirroring
    `chunking/`. Only `MeanPoolStrategy` (mean + L2 renorm) is
    registered.
  - `chunking/fixed_window.py` — **CLI default** long-doc chunker
    (option B from the catalogue). Tokenize → slice contiguous
    non-overlapping windows → decode each. Registered as
    `fixed-window`. Shipped at optimisation level L1 (tokenize +
    decode per chunk; `model.encode` re-tokenizes); L2/L3 deferred
    pending profiling evidence that tokenizer overhead matters.
  - `chunking/token_budget.py` — sentence-aware greedy packer (option
    D from the catalogue), registered as `token-budget` with a
    kwargs-capable factory. **Not the CLI default** after review —
    kept available via the registry for future A/B testing once
    `fixed-window` has a real-data baseline.
  - Chunking registry's `register_strategy` / `get_strategy` extended
    to forward kwargs (backward compatible — zero-arg factories
    still work).
  - `EncoderConfig.long_doc: LongDocConfig | None` field; when
    active, `TextBatcher` detects long docs, chunks them, encodes
    all texts in one batched `encode_texts` call (short-doc
    singletons riding the same batch as a long doc's K chunks),
    then aggregates per-record. Short-doc path is unchanged.
  - `is_long_doc(text, cfg)` helper with a cheap chars-per-token
    pre-filter to skip tokenisation on clearly-short docs.
  - Three new CLI flags: `--long-doc-strategy {truncate,chunk}`
    (default **`chunk`** — long docs are no longer silently
    truncated), `--long-doc-chunk-tokens N` (default **`None`** —
    auto-derived from `tokenizer.model_max_length -
    num_special_tokens_to_add(pair=False)` at CLI-init time; 8190
    for gte-multilingual-base; `_FALLBACK_CHUNK_TOKENS=8000` only
    when the tokenizer doesn't advertise the attrs),
    `--long-doc-aggregation {mean}`. `truncate` remains available
    as the opt-out for reproducing pre-step-16 outputs exactly.
  - Per-file telemetry: new `LONG_DOC_CHUNKED` counter.
  - Tests across
    `tests/test_aggregation.py`,
    `tests/test_chunking_fixed_window.py`,
    `tests/test_chunking_token_budget.py`,
    `tests/test_embed.py::TestTextBatcherLongDoc`,
    `tests/test_embed.py::TestIsLongDoc`,
    `tests/test_cli_create.py` (new-flag defaults, dry-run gating,
    aggregation-choices rejection, long-doc-config wiring,
    `_resolve_chunk_tokens` precedence). Full suite: 225/225 green,
    ruff clean.
- **Deferred** (each a small follow-up; the framework means most
  are a file + one `register_strategy` line + one `choices=` entry):
  - Other aggregation strategies: length-weighted mean, max pool,
    first-chunk, position-weighted, attention-weighted.
  - Other token-aware chunkers: fixed-window, stride overlap,
    paragraph packer, recursive, chonkie Token/Sentence (with
    verified tokenizer compat).
  - Upgrade the token-budget chunker to prefer `record["sents"]`
    when present (requires extending `ChunkingStrategy.chunk` or
    building the chunker per-record).
  - Regenerate the long-doc subset of any integration goldens to
    match the new chunk-aggregate default.
  - `n_chunks` metadata on `TextRecord` (gated on schema
    `additionalProperties` policy).
  - Calibrate `--long-doc-chunk-tokens` via a sweep on real long docs.
  - Calibrate `chars_per_token` per language for the fast gate.

**Full design-space discovery.** The notes doc retains the original
strategy catalogue (11 chunking options A–K, 8 aggregation options
α–θ), literature anchors, the deep-dive that settled "one-shot wins
for ≤8192-token docs because `gte-multilingual-base` is CLS-pooled",
and the set of OPEN items. Key takeaways still hold:

- BGE-M3's "chunk to ≤512" recommendation is a multi-vector-regime
  finding; it does not transfer to our single-vector text-level
  output.
- For docs ≤8192 tokens, one-shot encoding is the right choice, not
  chunk-then-pool (see "Deep dive" in the notes). The implemented
  boundary — "chunk only when >max_seq_length" — is principled.
- Late Chunking (Jina 2024) requires a mean-pooled model; it doesn't
  apply cleanly to our CLS-pooled encoder. Not deferred — rejected
  for this embedder. Reconsider only if we switch models.

**Validation.** The `--tol 1e-4` cosine validate contract still
holds — chunk-then-aggregate is deterministic, so a fresh golden
after flipping the default to `chunk` is a one-time cost. Regression
tests pass today (216/216). Acceptance on real data (recall on a
long-doc query set ≥ truncate-baseline) is deferred to real-GPU time;
record in the notes once measured.

**Backwards compat.** Long-doc documents at `--embedding-level text`
get different vectors under the new default (`--long-doc-strategy=chunk`)
than they did pre-step-16. The truncated behaviour was a bug (silent
data loss), not a feature, which is why `chunk` ships on. `truncate`
remains available for reproducing legacy outputs exactly. Short-doc
outputs are byte-identical — the short path was not touched.

Rationale, full strategy catalogue, rejected alternatives (Late
Chunking as a long-doc solution, paragraph-based packer, recursive
splitter via LangChain, hierarchical re-encoder), and the expected
Impresso-specific trade-offs in `.progress/long-doc-chunking/notes.md`.
