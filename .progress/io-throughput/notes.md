# IO-throughput — keeping the GPU fed

Companion to `.progress/gpu-throughput/notes.md` (GPU-side) and
`.progress/create-cli/notes.md` (which explicitly deferred prefetch/upload
overlap to this step). Goal: the CPU/IO side of the pipeline is not the
bottleneck during steady-state encoding on A100/H100.

## Current pipeline — where the stalls are

Per file, serially on the main thread:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  list → GET body → bz2.decompress → json.loads → rebuild_ft → batch →    │
│  encode (GPU) → bz2 write → upload                                        │
└──────────────────────────────────────────────────────────────────────────┘
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^            ^^^^^^^^^^
     CPU-bound / IO-bound — GPU idle                             GPU idle
```

Between files, everything before the first `encode()` of file N+1 is also
GPU-idle time: the boto3 GET of N+1 only begins after the upload of N
completes. On tens-of-MB-compressed / hundreds-of-MB-uncompressed yearly
shards, those dead zones add up.

Within a file, the same thread:
1. pulls lines from `bz2.open(Body)` (network + zlib-style chunked decode),
2. `json.loads` each line,
3. rebuilds full text from token offsets,
4. accumulates into `TextBatcher`,
5. on flush: calls `encode_texts` — this is the only step that feeds the GPU.

Steps 1–4 happen on the same thread as the CUDA launch loop. While they
run, the GPU has no work queued.

## Measurement is step 0, non-negotiable

Every benefit figure below is an order-of-magnitude estimate from published
benchmarks, not from our stack. `CLAUDE.md` calls for ≥85% SM utilization
on A100 during steady state; nothing emits that today. Before tuning,
instrument:

- **Per-stage wall time per file**: `download_s`, `decode_s`, `encode_s`,
  `write_s`, `upload_s`. A context manager + `log.info` is enough.
- **GPU SM utilization**: either a thread running `nvidia-smi dmon -s u`
  and tail-appending to the log, or `pynvml` sampled every 500 ms.
  Emit a single summary line per file: `gpu_util_mean`, `gpu_util_p10`.
- **Batch-flush cadence**: how long the GPU sits idle between encode
  calls. `time.monotonic()` around `encode_texts` vs. around the outer
  iterator. If flush-to-flush > encode duration, IO is the bottleneck.

Only then do the decisions below get made with numbers.

## Ideas, classified by benefit × complexity

Benefit is the expected end-to-end throughput delta if the stage is
currently on the critical path. Complexity is rough LOC + review surface.
**All tier assignments are hypotheses pending measurement.**

### Tier A — low complexity, likely-high benefit

**A1. Instrumentation (prerequisite, see above).**
Without it, everything else is guessing. ~40 LOC, no new deps.

**A2. `orjson.loads` / `orjson.dumps` in place of stdlib `json`.**
Drop-in. Published figures: `orjson.loads` ≈ 2–6× stdlib, `orjson.dumps`
≈ 10× stdlib on realistic payloads with mixed types and small-medium docs
[[orjson README](https://github.com/ijl/orjson), [dollardhingra benchmark](https://dollardhingra.com/blog/python-json-benchmarking/)].
Special leverage on our *output* side: each embedding record contains a
list of ~768 floats; `orjson` with `OPT_SERIALIZE_NUMPY` can serialize the
underlying numpy buffer directly, reported at 4–12× stdlib and ~1/3 the
memory [[orjson README](https://github.com/ijl/orjson)]. Requires keeping
vectors as `np.ndarray` (not `.tolist()`) until write. Small change in
`embed.py` and `pipeline.py`.

**A3. Prefetch file N+1 while encoding file N.**
The deferred item from `.progress/create-cli/notes.md`. Single background
thread driven by a `queue.Queue(maxsize=1)` of `(input_key, local_tempfile)`
tuples. While the main loop encodes N, the prefetcher has already issued
the GET for N+1 and possibly finished decompressing it to a local spool
file. Stdlib `bz2` releases the GIL on read [[Python c-api/threads](https://docs.python.org/3/c-api/threads.html)],
so the prefetch thread actually progresses while the main thread runs
CUDA launches. ~60 LOC, handles the between-files dead zone.

**A4. Async upload of completed output.**
Hand the finished local tempfile to a `ThreadPoolExecutor(max_workers=1)`
and move on. Errors surface at the next `.result()` call or at provider
loop exit. ~20 LOC. Pair with A3 — both use the same executor pattern.

**A5. boto3 `TransferConfig` for multipart ranged downloads.**
The current code streams `Object.get()['Body']`, a single TCP stream.
`s3.download_file` / `download_fileobj` with
`TransferConfig(max_concurrency=10, multipart_chunksize=8*1024*1024)` issues
parallel ranged GETs and assembles locally [[boto3 s3 guide](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html)].
Ceph RADOSGW (Switch Engines' backend) is S3-compatible including Range
requests and multipart [[Ceph S3 API docs](https://docs.ceph.com/en/latest/radosgw/s3/)].
Trade-off: breaks streaming. The whole compressed file lands on local
disk before `bz2.open` starts. Only worth it if bandwidth per TCP stream
is the limiting factor — verify via A1 before adopting. Pairs naturally
with A3 (the prefetcher writes to a local spool anyway).

### Tier B — moderate complexity, likely-high benefit — **NOT IMPLEMENTED**

*Status: none of the Tier B items below shipped in step 13. They remain
candidates for a future step; revisit once real-A100/H100 measurement
(from the A1 instrumentation that did land) points at a specific
bottleneck these would relieve.*


**B1. Parallel bz2 decompression with `indexed_bzip2`.**
`indexed_bzip2` (same author as `rapidgzip`) does multi-threaded bzip2
decompression. Reported ~6× speedup on a 12-core machine vs stdlib `bz2`,
but **37% slower than stdlib on a single core** [[indexed_bzip2 README](https://github.com/mxmlnkn/indexed_bzip2)].
So it's only a win on the container if we actually have spare cores —
the Run:AI pod request should ensure that. API is file-like and accepts
a readable; should be drop-in for `bz2.open(body, mode="rt")`. Adds a
dep. High leverage because bz2 is the classic bottleneck of this format
(decompress ~20–40 MB/s single-threaded; `pbzip2`/`indexed_bzip2` scale
roughly linearly up to ~8 cores).

**B2. In-file reader/decoder thread.**
One thread does `GET → bz2 → json → rebuild_ft → push to queue`; main
thread pops batches and encodes. Overlaps steps 1–4 above with GPU work
even within a single file. The GIL cost is limited because boto3 socket
reads and bz2 C decompression both release it, and orjson is fast enough
that the JSON stage is no longer a dominating GIL holder. ~120 LOC.
Subsumes A3 for the within-file case; run both together.

**B3. Output codec: bz2 → gzip or zstd on the output side.**
Input format is set by upstream Impresso and stays `bz2`. Output is our
choice. Published compression throughput on realistic data: bz2 ~11 MB/s,
gzip ~16 MB/s, zstd (default) ~132 MB/s
[[Broadcom ZSTD metrics](https://knowledge.broadcom.com/external/article/404122/zstd-compression-metrics.html),
[Manish Jain compression comparison](https://manishrjain.com/compression-algo-moving-data)].
On the output side we write one line per record, each containing a 768-d
float list as text — hundreds of KB per file, not GB, so the absolute
time is small. Most likely small win unless the embedding-level is
`sentence` or `chunk` where record count is 10–100× higher. Blocker: any
downstream consumer expects `.jsonl.bz2`. Needs a decision outside this
repo before adopting. Rename to `.jsonl.zst` or keep the extension and
hope nothing breaks — the former is right.

**B4. Token-budget batching instead of record-count batching.**
`TextBatcher` currently flushes every `batch_size` records. Record length
varies wildly: a 400-char item and an 80 000-char item both count as "1".
Switch to a token-budget flush (e.g. 32 000 tokens per batch, configurable),
measured with the tokenizer's `encode`. Sentence-transformers' `encode()`
already sorts inputs by length internally (`length_sorted_idx = np.argsort([len(s) for s in sentences])`,
source: [`SentenceTransformer.py`](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py)),
so intra-batch padding is already minimized — the win here is *inter-batch*
uniformity: fewer tiny batches for short items, and no OOM risk from
unexpectedly packing giant items together. ~80 LOC in `embed.py`.

### Tier C — higher complexity, selective benefit — **NOT IMPLEMENTED**

*Status: none of the Tier C items below shipped. All depend on a Tier B
item landing first or on measurement showing a specific sub-stage as the
residual critical path.*


**C1. `msgspec.json.Decoder[InputSchema]`.**
With a typed schema, `msgspec` decodes faster than `orjson` and uses 6–9×
less memory on large files [[pythonspeed](https://pythonspeed.com/articles/faster-python-json-parsing/),
[msgspec benchmarks](https://jcristharif.com/msgspec/benchmarks.html)].
Memory matters more than speed for us: a single parsed line with
thousands of sentences of token-offset dicts allocates a lot of dict
objects; `msgspec.Struct` is a C-level tuple-backed type. Cost: define
the input schema explicitly (which we'd want anyway — see `schema.py`).
Without a schema, `msgspec` is on-par with `orjson`, so this is only
worth it after A2 is proven insufficient.

**C2. Pre-tokenize on CPU while GPU encodes previous batch.**
The `AutoTokenizer` fast-path is Rust and releases the GIL. Running
tokenization of batch N+1 on a CPU thread while the GPU runs the forward
pass of batch N overlaps the two. Sentence-transformers does tokenization
inside `encode()`, so this requires calling the tokenizer ourselves and
feeding tensors to the underlying transformer — i.e., bypassing part of
`SentenceTransformer.encode`. Non-trivial; defer until B2 is in place
and measurement shows tokenization as the residual bottleneck.

**C3. Chunker pre-compute on a worker thread (chunk-level only).**
`chonkie.SemanticChunker` uses its own small embedder and segmenter to
split long documents. That's a mini-pipeline per record. Running chunking
for record N+1 on a worker thread while the GPU embeds chunks of record N
overlaps them. Only matters for `--embedding-level=chunk`. ~60 LOC.

### Deferred / rejected

- **Multi-process pipeline.** Threads suffice because the stages that
  would benefit from process parallelism (bz2, boto3 I/O, tokenizer)
  already release the GIL. Process parallelism pays serialization cost
  at every boundary and doesn't help the single GPU we feed.
- **`aioboto3` / pure asyncio pipeline.** With at most 1–2 files in flight
  and no fan-out, async's cognitive cost outweighs the benefit. Threads
  are the right tool here.
- **Batching encode calls across files.** Breaks the 1:1 input→output
  contract in `CLAUDE.md`. Not considered.
- **Upstream format change bz2 → zstd at rest.** Cross-team, long-lead.
  Tracked as a separate conversation with the Impresso pipeline team, not
  this repo.
- **In-memory block cache / fsspec caching.** We don't re-read files;
  caching adds memory pressure for no hit-rate.
- **Streaming multipart uploads.** Covered in `.progress/io-layer/notes.md`:
  local tempfile + single-call upload is simpler and correct at our sizes.

## Recommended sequence

1. **A1** (measurement) first. Land a week of real-run logs.
2. If GPU idle time between files dominates: **A3 + A4** (prefetch + async
   upload). Smallest diff, biggest expected lift.
3. If bz2 decompress dominates inside a file: **B1** (`indexed_bzip2`)
   and/or **B2** (in-file reader thread). B1 alone may be enough.
4. If JSON parse/serialize dominates: **A2** (orjson), then **C1**
   (msgspec) only if memory becomes the constraint.
5. **A5** (boto3 `TransferConfig`) only once the rest is in place — it
   changes the streaming model and is wasted effort if bz2 decompress is
   still the critical path.
6. **B3** (output codec) is a cross-team decision, not a code change;
   surface it separately.
7. **B4, C2, C3** are second-order — revisit when measurement says the
   first-order stages are no longer dominant.

## Open questions — fill in after A1

- Actual distribution of per-stage times on A100 and H100. Record here
  as a small table.
- Whether the Run:AI pod's CPU request gives us the 4+ cores that make
  B1 worthwhile; if not, adjust the pod spec first.
- Whether Switch Engines' Ceph RadosGW supports the multipart GET path
  at the bandwidth we'd actually want. Confirm with a one-shot benchmark
  (`s3.download_file` with `max_concurrency=10` vs. streaming `Body`) on
  a real shard and record the MB/s.
- Whether keeping embeddings as `np.ndarray` through write (for A2's
  `OPT_SERIALIZE_NUMPY`) is compatible with our 5-decimal rounding;
  orjson's numpy path emits full precision by default, so we may need to
  pre-round with `np.round(v, 5)` before passing in.

## Landed (Tier A)

Shipped in step 13 on 2026-04-22. The hypothesis-tier classification above
survives; what follows records the concrete choices so a future session can
tell what is "actually wired" vs "still in the idea catalogue".

### A1 — instrumentation (`telemetry.py`)

- `StageTimer.stage(name)` context manager accumulates wall time per stage
  into a dict. Used in `pipeline.process_file` and `pipeline.process_provider`
  to measure `download`, `encode`, `upload_wait`.
- `GpuSampler` is a context manager wrapping a background thread at 2 Hz
  that calls `pynvml.nvmlDeviceGetUtilizationRates`. `pynvml` is
  **soft-imported inside `__enter__`** — missing package, no CUDA, or init
  failure all degrade to "zero samples, empty summary". No dependency on
  `pynvml` is declared in `pyproject.toml` because NGC `pytorch:25.03-py3`
  already ships it.
- `format_stats_line(prefix, timer, records=n, gpu=summary)` writes the
  per-file INFO line at end-of-file. Omits the `gpu_util_*` block when
  there are no samples (keeps local CPU-only dev logs clean).
- `GpuSampler` uses a ring-buffer-less list (`self._samples`) and publishes
  the window via `summary()`. The summary is delivered back to
  `process_provider` via a module-level dict keyed on `input_key.key`; this
  avoids threading an extra return value through `_encode_to_local`.

### A2 — orjson on both sides

- Input parsing: `_iter_parsed` and `validate.parse_records` both use
  `orjson.loads`. One behaviour change documented in CLAUDE.md: orjson is
  strict RFC 8259 and rejects bare `NaN` at parse time, so files containing
  NaN embeddings now surface as "malformed JSON" errors rather than
  "non-finite embedding" errors. Either way they fail validation — test
  `test_catches_nan` was updated to assert "errors is non-empty".
- Output writing: `orjson.dumps(out, option=orjson.OPT_APPEND_NEWLINE)`
  returns bytes with a trailing `\n`. The output bz2 file is therefore
  opened in binary mode (`bz2.open(path, "wb")`). Embedding rounding still
  happens in `schema._round_embedding` on the list-of-floats path; the
  `OPT_SERIALIZE_NUMPY` fast-path is **not** wired yet because embeddings
  reach `orjson.dumps` as Python lists (via `.tolist()` inside
  `TextRecord.to_dict`). That's a deferred micro-optimisation — moving the
  round+encode to operate on the `np.ndarray` directly would unlock the
  4–12× numpy fast-path, but also requires threading `np.round` through
  each item-dict before serialization. Left for a later round once
  measurement shows `dumps` on the critical path.

### A3 + A4 — prefetch + async upload

- `process_provider` is rewritten around two `ThreadPoolExecutor(max_workers=1)`
  instances, `prefetch` and `upload`. At most one download + one encode
  + one upload are in flight at any time.
- Skip logic runs **before** prefetching (no point downloading a file we'd
  discard). `_plan_files` returns the `(input_key, output_key)` pairs that
  will be processed; `_dry_run_summary` is a separate code path so the
  executor setup never runs under `--dry-run`.
- Backpressure: before submitting upload N+1, the loop awaits
  `prev_upload.result()`. A failed upload raises there and propagates —
  no "silent drop" behaviour.
- `process_file` keeps the sequential path (download → encode → upload)
  for direct callers and tests. It now uses `io.download_to_local` +
  `io.iter_jsonl_bz2_path` instead of the streaming `iter_jsonl_bz2` GET.
- Regression guard: `test_process_provider_overlaps_prefetch_and_upload`
  uses two `threading.Event`s to pin file N's upload mid-flight and
  asserts that file N+1's download has already started by then.

### A5 — boto3 `TransferConfig` multipart GET

- `io.DEFAULT_TRANSFER_CONFIG = TransferConfig(multipart_threshold=8 MB,
  multipart_chunksize=8 MB, max_concurrency=10, use_threads=True)`.
- `io.download_to_local(bucket, key, dest, transfer_config=None)` drives
  `s3r.Bucket(bucket).download_file(key, dest, Config=cfg)`. Whole
  objects land as a local file before `bz2.open` reads them — this
  intentionally gives up streaming in exchange for parallel ranged GETs.
  Acceptable because the prefetcher already decouples download from
  encode, so the latency hit from materialising the whole compressed
  file is hidden behind the previous file's encode.
- The constant is exposed so callers (future benchmarks or tuned CLI
  flags) can construct an alternate config without reaching into boto3.

### Known deferred from Tier A — **NOT IMPLEMENTED**

*Sub-items within the Tier A scope that didn't ship. Re-promote as their
own tasks if measurement justifies it.*

- **`orjson.OPT_SERIALIZE_NUMPY` fast path** on the output side — requires
  refactoring `_round_embedding` to operate on numpy arrays before
  serialization. Deferred until measurement shows `dumps` on the critical
  path.
- **Configurable `TransferConfig` via CLI flag** (`--transfer-concurrency`,
  `--multipart-chunksize`) — not wired. The defaults are sensible for
  Switch Engines; revisit if measurement suggests a different tuning.
- **CLI flag to disable the prefetch+upload overlap** — useful for
  bisecting regressions in the field. Add if the pipelined loop ever
  becomes suspect.

## Sources

- [boto3 — file transfer configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html)
- [boto3 issue #3466 — multipart range download limitation](https://github.com/boto/boto3/issues/3466)
- [Ceph RadosGW — S3 API](https://docs.ceph.com/en/latest/radosgw/s3/)
- [indexed_bzip2 — parallel bzip2 for Python](https://github.com/mxmlnkn/indexed_bzip2)
- [orjson — fast Python JSON + numpy](https://github.com/ijl/orjson)
- [msgspec — typed JSON decode + benchmarks](https://jcristharif.com/msgspec/benchmarks.html)
- [pythonspeed — faster Python JSON parsing with msgspec](https://pythonspeed.com/articles/faster-python-json-parsing/)
- [dollardhingra — json vs ujson vs orjson benchmarks](https://dollardhingra.com/blog/python-json-benchmarking/)
- [Broadcom — ZSTD compression metrics](https://knowledge.broadcom.com/external/article/404122/zstd-compression-metrics.html)
- [Manish R Jain — compression algorithms benchmark](https://manishrjain.com/compression-algo-moving-data)
- [Python c-api/threads — GIL release around blocking I/O](https://docs.python.org/3/c-api/threads.html)
- [sentence-transformers SentenceTransformer.py — length_sorted_idx](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py)
