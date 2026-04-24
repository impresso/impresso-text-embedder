# Structured logging — design notes

## What shipped

Two-handler setup, wired from `cli/create.py`'s `main()`:

- **File handler.** One file per provider per day at
  `/rcp-scratch/<username>/experiments/embeddings/<YYYY-MM-DD>/<provider>.log`.
  Level configurable via `--log-level-file` (default `INFO`). Captures
  everything the old run emitted: per-file `format_stats_line` telemetry,
  skip/reprocess decisions, GPU profile detection, the final
  processed/skipped summary. Grep-friendly format
  `%(asctime)s %(levelname)s %(name)s: %(message)s` — same as before.
- **Terminal handler.** `TqdmLoggingHandler` at level `ERROR`, routes via
  `tqdm.write(...)` so a stack trace lands above the progress bar instead
  of shredding it. Nothing else reaches the terminal through the logging
  system.
- **Terminal progress.** `process_provider` wraps its `to_process` loop
  in `tqdm.tqdm(total=N, file=sys.stderr, disable=None)`.
  `set_description("alias/year")` on file start; `set_postfix(dl=…s
  enc=…s up=…s gpu=…%)` after each completed file using the already-
  populated `StageTimer.totals` and `GpuSummary`. `disable=None` makes
  the bar no-op on non-tty (pytest, piped output), so the existing test
  suite needs no tqdm mocking.
- **Bracketing prints.** The resolved log path prints to stderr with a
  plain `print(..., file=sys.stderr)` before any logger/tqdm activity
  ("logging to /rcp-scratch/..."), and the final processed/skipped
  summary prints the same way after the bar closes. These use `print`
  rather than logging so they show regardless of handler levels and
  never collide with the bar.

## Default path — rationale

`/rcp-scratch/<username>/experiments/embeddings/<YYYY-MM-DD>/<provider>.log`
is the production path. Rationale:

- **Absolute path, not `$HOME`-relative.** The Run:AI PVC mounts at
  `/rcp-scratch` (see `Makefile: RCP_SCRATCH_PATH`). `$HOME` inside the
  container is an ephemeral overlay and vanishes at pod exit. Scratch
  persists across jobs.
- **Username via `getpass.getuser()`.** Inside the container, the user
  is the LDAP account the Dockerfile creates from build args
  (`LDAP_USERNAME`). `getpass.getuser()` resolves to the same string,
  which is all we need — the Python code doesn't need to know
  *`LDAP_USERNAME`* as a build-time constant, just the resolved name at
  runtime. This also means local dev runs (without a container) write
  under the developer's own username, which is correct.
- **Date directory, not per-run.** We considered
  `<YYYY-MM-DDTHHMMSS>/<provider>.log` to isolate reruns. Rejected:
  in practice RCP jobs are idempotent re-runs of the same provider on
  the same day (checkpoint behaviour), and a single appended log is
  easier to search than a fan-out of timestamp directories. Each log
  line carries an ISO-8601 timestamp in the format string, so
  distinguishing runs after the fact is grep-of-a-day, not
  find-of-a-directory.
- **One file per provider, not per file.** Per-file logs would bury the
  cross-file narrative (prefetch overlap, skip decisions, the final
  summary). Per-provider keeps related output together.

## `/rcp-scratch` missing → fail fast

If `/rcp-scratch/` does not exist and no `--log-dir` override is given,
the CLI exits with a clear message asking the caller to mount the PVC
or pass `--log-dir`. Three rejected alternatives:

1. **Silent fallback to `./logs/`.** Rejected: masks a PVC misconfig on
   the RCP side. A job that "ran" but wrote its log to an ephemeral
   container path and then lost it is worse than a job that refused to
   start. The error message names both remedies, so a dev can pass
   `--log-dir /tmp/logs` and be unblocked in one flag.
2. **Fallback to `$HOME`.** Rejected for the same reason —
   container `$HOME` is ephemeral; local `$HOME` makes log locations
   differ between dev and prod.
3. **Auto-detect "am I in a container?" and branch.** Rejected as
   brittle. `--log-dir` is explicit, one flag, and covers every case.

The existence check is `Path("/rcp-scratch").is_dir()`, not a write
probe — permissions errors at file-open time surface through the normal
`OSError` path.

## Why tqdm + ERROR-only on the terminal

The previous behaviour — INFO streaming to stderr — scrolls far too
fast to read during a 50-file provider run, and buries the signal the
submitter actually cares about (how many files are left, is the GPU
warm). The bar gives a single live line with throughput per file; the
file handler keeps the full history for post-hoc debugging.

ERROR is the only level that breaks through to the terminal because
ERRORs are rare but actionable (an S3 failure mid-run, a validation
mismatch) — the submitter wants to see them without waiting for the
job to finish. `tqdm.write` is the right primitive: it writes a line
above the bar and redraws, so the bar stays intact.

## tqdm postfix choice

Postfix shows the *just-completed* file's telemetry, not a rolling
average, because:

- `StageTimer.totals` is already the per-file dict, no extra
  bookkeeping.
- Per-file numbers expose outliers (a slow upload) that a running mean
  would smooth over.
- The values are self-explaining in abbreviated form: `dl=12s
  enc=45s up=2s gpu=92%`. Fits in a terminal row next to the bar.

GPU mean is shown without p10 to keep the postfix short; the file log
retains the full `gpu_util_mean=… p10=… n=…` line.

## Scope — `impresso-embed-validate` unchanged

Validate is a one-shot diagnostic (stderr is fine, no file logging),
and adding a tqdm bar over a single file makes no sense. Left
untouched. If a future use case needs the same treatment, lifting
`configure_logging` into `logging_setup.py` makes that a one-line
change.

## Alternatives rejected

- **Structured JSON logs.** Would be nicer to grep with `jq`, but we
  don't have a log-aggregation consumer yet. Keep the familiar
  `asctime level name: message` format; revisit if we get one.
- **Rotating file handler.** Not worth the complexity for our volume
  (a provider run produces dozens of lines per file × ~50 files = a few
  thousand lines). The per-day file will not hit a size limit in any
  realistic run.
- **`rich.progress` instead of `tqdm`.** `rich` is heavier and its
  progress primitives are less established; `tqdm` is one already-
  familiar dep. No strong reason to switch.

## Known gaps / follow-ups

- **No log rotation or retention policy.** If scratch fills up, ops
  will need to clean `/rcp-scratch/<user>/experiments/embeddings/`
  manually. Adding `find … -mtime +30 -delete` to a weekly job is a
  cheap follow-up once there's an ops owner.
- **`impresso-embed-validate` not covered.** Deliberately out of scope
  (see above). Reopen if someone runs validate at scale.
- **No per-stage GPU util.** `GpuSampler` only wraps the encode stage;
  the postfix reports encode-stage mean. That's what we want today but
  if we ever profile upload-GPU overlap we'll need per-stage sampling.
