# Multi-GPU sharding — file-level data parallelism

## TL;DR

`--shard-index i --num-shards N` on `impresso-embed-create`, round-robin
selection over `list_objects_v2`'s lexicographic output, applied lazily as
a filter at `pipeline._plan_files`. One runai job per shard, one GPU per
job, model replicated across jobs. No coordination between shards beyond
the static `(i, N)` partition. Lifts CLAUDE.md → Non-goals "Multi-GPU,
multi-node, DDP." for the data-parallel case only — model-parallel
mechanisms (DDP, FSDP, tensor-parallel) remain out of scope.

## Problem

The migration runs one `runai-submit` job per provider on one A100/H100
node. The 305M-param `gte-multilingual-base` fits 5–10× over on either
GPU, so model-parallel mechanisms (DDP, FSDP, tensor-parallel) solve a
problem we don't have. What we want is **horizontal throughput**: split a
provider's file list into N disjoint partitions and run one runai job per
partition. Each shard is fully independent; failures recover by
re-submitting the same `(i, N)` job.

The constraint to amend in CLAUDE.md is one bullet under Non-goals:
"Multi-GPU, multi-node, DDP." This step lifts the multi-GPU bar
specifically for the data-parallel case (replicated model, partitioned
files, one container per shard, one GPU per container) — multi-node and
model-parallel mechanisms remain out of scope.

## Decision

**`--shard-index i --num-shards N` on `impresso-embed-create`,
round-robin selection (`enumerate(keys) % N == i`) over the listing
returned by `s3io.list_input_keys`.** The flag pair contract is borrowed
from pytest-shard and HuggingFace `datasets.shard(num_shards, index,
contiguous=False)` — same names, same semantics. Defaults
(`--shard-index 0 --num-shards 1`) reproduce today's single-job behaviour
bit-for-bit.

The filter lives in `pipeline._plan_files` (right after the
`s3io.list_input_keys(...)` call), not inside `list_input_keys` itself.
Keeps the S3 lister single-purpose and reusable for
`impresso-embed-validate`.

## Why round-robin specifically

Five candidate schemes:

| Scheme | Picked? | Reasoning |
|---|---|---|
| **Round-robin** (`k % N == i`) | ✓ v1 | Dead simple, deterministic, no salt issues. Spreads sorted skew: keys sort by `<alias>-<year>`, so old years (small) interleave with new years (dense) across shards. |
| Contiguous range (`keys[i*L//N : (i+1)*L//N]`) | ✗ | Equally simple, but with year-sorted keys gives shard 0 all the small old shards and shard N-1 all the dense modern ones. Real wallclock imbalance. |
| Stable hash mod (`int(sha1(key)[:16], 16) % N == i`) | ✗ | "Stable when N changes" doesn't matter — N is fixed per run, and `--skip-if-s3-exists` covers re-runs. Adds a hash dependency for no payoff. |
| Size-aware greedy (sort by `obj["Size"]` desc, drop into least-loaded bin) | Deferred | Best balance, but file size ≠ encode time (compression ratio + doc count both vary). Land round-robin first; revisit if measured shard-wallclock skew is >2× on real data. Adding `size` to `InputKey` is a 1-line change in `io.py:77–82` + `io.py:167`. |
| Consistent hashing | ✗ | Solves elastic resharding with minimal data movement. Overkill — N is fixed for the run; we don't add/remove GPUs mid-job. |

## Round-robin under monotonic file sizes (impresso data)

Impresso file sizes grow roughly monotonically with year (later years
have more content than early years), which is the *favourable* case for
round-robin and tightens the size-vs-encode-time argument too:

1. Round-robin is already decent on monotonic data — interleaving a
   sorted sequence is a classic balanced-partition trick. With sizes
   growing roughly monotonically and round-robin over the lex listing,
   each shard gets a mix of early-tiny and late-fat files. Worst-case
   ratio is bounded around 2× (vs. ~5× for chronological blocking).
2. Size-weighted greedy is then strictly better and easy — sort desc,
   drop each file into the currently-smallest shard. On monotonic data
   this approaches 1× perfect balance. The "size-doesn't-track-encode-time"
   worry I raised before is also weaker for you: same-newspaper-different-year
   content is fairly homogeneous, so size, record count, and total chars
   all correlate well.
3. Cost is still zero S3 calls — `list_objects_v2` already returns `Size`.

What this *doesn't* change: greedy bin-packing still loses the
deterministic-re-submit property (assignments depend on the full set of
sizes, so adding/removing a year reshuffles), and on the current cost
model (RCP releases each GPU as the shard finishes; no idle-GPU bill)
wallclock skew has no $-cost. So the analysis upgrades the *theoretical*
case for greedy on impresso data but doesn't change the recommendation:
keep round-robin until something downstream actually pushes back on
end-to-end wallclock.

## Mechanism — where the shard filter lives

`list_objects_v2` returns keys in [UTF-8 lexicographic
order](https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html);
no explicit sort needed. Ceph RadosGW (Switch Engines) inherits this
contract. Apply round-robin lazily as the listing iterator is consumed.

**Insertion point**: `pipeline.py:_plan_files` (line 270), after
`s3io.list_input_keys(...)`:

```python
listing: Iterable[s3io.InputKey] = s3io.list_input_keys(...)
if cfg.num_shards > 1:
    listing = (
        k for i_k, k in enumerate(listing)
        if i_k % cfg.num_shards == cfg.shard_index
    )
if limit is not None:
    listing = itertools.islice(listing, limit)
```

`limit` applies *after* sharding so `--limit 10 --shard-index 2
--num-shards 4` means "10 files of the shard-2 partition", not "10 files
then partition".

**No re-grouping needed.** `cli/create.py:28` makes `--provider` strictly
single-valued (`required=True`, no `nargs`), so each invocation already
operates on one provider. The shard filter narrows the set of files
within that provider; `process_provider`'s prefetch+upload overlap
(`pipeline.py:346–480`) keeps working unchanged on the narrower file set.

## CLI surface

`cli/create.py`:

- `--shard-index INT` (default `0`): zero-indexed shard this job owns.
- `--num-shards INT` (default `1`): total shards in this run.
- Validation at parse time: `num_shards >= 1`, `0 <= shard_index <
  num_shards`. Fail loudly with the offending values quoted —
  `--shard-index 4 --num-shards 4` should not silently process zero
  files. Both flags or neither — error if exactly one is supplied with a
  non-default value (use a sentinel default and check together).

`PipelineConfig` in `pipeline.py`: gain `shard_index: int = 0` and
`num_shards: int = 1`. Default values reproduce single-job behaviour.

## Robustness invariants

1. **Deterministic ordering**: `list_objects_v2` returns lexicographic
   order per the AWS S3 contract; Ceph RadosGW inherits it. Load-bearing
   — if either changes, partitions become non-disjoint.
2. **Idempotent re-run**: same `(--shard-index, --num-shards)` → same
   files → `--skip-if-s3-exists` + the `reembed-on-change`
   `LastModified` check handle "what's already done". A failed shard
   re-runs by re-submitting the same runai job.
3. **No coordination**: no S3 lock files, no leader election, no
   work-stealing. Each job decides from `(i, N)` only.
4. **Manifest at startup**: one INFO line per run —
   `shard 2/4: 47 files, first=…/foo-1850.jsonl.bz2
   last=…/bar-1999.jsonl.bz2`. `runai describe job` then suffices to
   audit which files a shard was *supposed* to run.
5. **Per-shard log path**: `logging_setup._resolve_log_path`
   (`logging_setup.py:47–78`) gains optional `shard_index` /
   `num_shards` kwargs; when present, filename becomes
   `<provider>-shard-<i>-of-<N>.log` in the same date directory. Without
   them the path is unchanged
   (`<log_dir>/<YYYY-MM-DD>/<provider>.log`).
6. **No silent fallback**: if exactly one of `--shard-index` /
   `--num-shards` is supplied with a non-default value, error at parse
   time. Both or neither — never one.

## Makefile shape

Two new targets, no Python in the Make layer (it stays a `docker
buildx`/`runai-submit` wrapper per CLAUDE.md):

```makefile
# Submit one shard. SHARD_INDEX and NUM_SHARDS required.
runai-submit-shard:
	runai submit ${JOB_NAME}-shard-${SHARD_INDEX}-of-${NUM_SHARDS} \
		--image ${IMAGE} \
		... \
		-- --provider ${PROVIDER} \
		   --shard-index ${SHARD_INDEX} \
		   --num-shards ${NUM_SHARDS} ${EXTRA_ARGS}

# Loop over 0..N-1 and submit each shard as a separate runai job.
runai-submit-multi:
	@for i in $$(seq 0 $$((${NUM_SHARDS} - 1))); do \
		$(MAKE) runai-submit-shard SHARD_INDEX=$$i NUM_SHARDS=${NUM_SHARDS}; \
	done
```

`make runai-submit-multi NUM_SHARDS=4 PROVIDER=BNF` then yields four
independent runai jobs. RCP scheduler decides node placement. No
container-side multi-GPU plumbing.

## Rejected alternatives

- **Intra-container `torch.multiprocessing.spawn` (one runai job, N
  GPUs)**. Smaller scheduling footprint and lets one process aggregate
  telemetry, but binds N GPUs to a single pod — RCP's queue tends to
  schedule 1×GPU pods faster than N×GPU pods, and the existing
  one-image-one-GPU contract is simpler to debug. Reconsider only if RCP
  scheduling actually delivers N×GPU pods within reasonable wait times.
- **DDP / FSDP / tensor-parallel**. Solve model-parallel problems we
  don't have (gradient sync, model-too-big-for-one-GPU). Adds NCCL on a
  hot path that needs no cross-GPU traffic.
- **`accelerate launch` / HF Accelerate**. Pure overhead — `accelerate`'s
  value is launcher abstractions for DDP/FSDP/notebook, none of which
  apply here.
- **vLLM / Triton Inference Server**. Designed for streaming inference
  servers with batching. We have offline file-batch encoding; the model
  isn't even decoder-style.
- **Shared S3 work queue** (each job claims files via S3 lock objects).
  Adds a coordination dependency for a problem already solved by static
  partitioning. The `--skip-if-s3-exists` idempotency contract already
  covers crash-recovery without locks.
- **Provider-level sharding** (assign whole providers to jobs). Coarser,
  simpler at first glance, but provider sizes vary by 10× or more — one
  job ends up doing all the work. File-level sharding is finer-grained
  at no extra complexity.
- **Shard filtering inside `list_input_keys`**. Keeps the lister
  single-purpose; sharding is a pipeline-orchestration concern.
  Filtering in `_plan_files` keeps `list_input_keys` reusable for
  `impresso-embed-validate`.
- **`--shard-spec '2/4'`-style single flag**. Two integer flags match
  pytest-shard / HF datasets / GitHub Actions matrix; one combined string
  flag would be novel for no win.

## Open items

- **Calibration**: measure shard wallclock skew on a real provider with
  N=4 on RCP. If skew is >2× across shards, promote the size-aware
  greedy scheme (size already returned by `list_objects_v2`; `InputKey`
  gains a `size` field; partitioning runs on the materialized list). If
  skew is <1.5×, leave round-robin as is. Acceptance bar: a 4-shard run
  on the largest provider where the slowest shard finishes within 1.5×
  the fastest's wallclock.
- **Telemetry across shards**: per-shard logs land in separate files;
  combining them into a single per-run summary (e.g. cumulative
  `gpu_util_mean` across N shards) is an orchestration-side aggregator
  task, not a code change here.
- **Bisect helper**: a `--shard-bisect` flag that, given a failing
  `(i, N)`, runs the shard's first half then the second half. Defer
  until something actually breaks shard-only.
- **Per-node-type batch size**: today the per-profile default in
  `accel.py` applies. If different RCP nodes give different capabilities
  (A100-40GB vs A100-80GB) within one multi-shard run, the per-job
  autodetect handles it — but worth verifying after the first 4-shard
  real run.
- **Cross-shard provider load balancing**: provider sizes vary
  drastically. If one provider needs 8 shards and another needs 1, that's
  an operator decision per-provider, not something the code coordinates.
  Document the recipe in the README rather than building it in.

## Sources

- [pytest-shard — round-robin sharding by `--shard-id` / `--num-shards`](https://github.com/AdamGleave/pytest-shard)
- [HuggingFace `datasets.shard(num_shards, index, contiguous=False)`](https://huggingface.co/docs/datasets/process)
- [Bazel test sharding — deterministic shard assignment in CI](https://vincerose.dev/posts/bazel-test-sharding-detail-python/)
- [AWS S3 ListObjectsV2 — UTF-8 lexicographic order guarantee](https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html)
- [Consistent hashing — when stable assignment matters (and when it doesn't)](https://en.wikipedia.org/wiki/Consistent_hashing)
- [HuggingFace — Distributed Inference (Accelerate)](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference)
- [PyTorch — Multiprocessing best practices (spawn + CUDA)](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [Bin-packing for load distribution — when round-robin isn't enough](https://developer.ibm.com/articles/mastering-optimization/)
