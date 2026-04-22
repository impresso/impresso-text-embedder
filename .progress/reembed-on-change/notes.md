# Re-embed on input change — decisions

## Problem

The `feat/migration-python-package` migration dropped the old Make/stamp tree. With it we lost mtime-driven reprocessing: when a `.jsonl.bz2` shard is re-uploaded on S3, the new code has no way to notice because `process_file` only checks whether the output *exists*.

`impresso_essentials.versioning` is the long-term answer (per-stage `DataManifest` JSONs, diffed across runs). It's also a larger commitment than this project needs today. We want a narrow fix that covers the common case — operator re-ingests a shard — without building the full manifest loop.

## Decision

**Compare S3 `LastModified` at skip-decision time.** Two states:

| `--force` | output exists? | input newer than output? | action |
|---|---|---|---|
| yes | any | any | reprocess |
| no | no | — | reprocess |
| no | yes | no (output ≥ input) | skip ("up-to-date") |
| no | yes | yes (output < input) | reprocess ("input newer") |

No new CLI flag. Existing `--force` is the escape hatch. The logs distinguish the two skip/reprocess reasons so operators can tell why a file was picked up.

## Why not the other options

The earlier exploration laid out four options; the trade-off picked here is (1).

- **(2) Input ETag stored in the output header.** Exact (byte-level), but needs a schema bump and a one-line read of every existing output to decide skip. Throwaway once versioning lands.
- **(3) Sidecar manifest per output.** Same information as (2) in a separate file. Extra PUT/HEAD per file and still throwaway.
- **(4) `impresso_essentials.versioning` manifest.** The proper answer. Requires the upstream stage to also emit a manifest, a `DataStage.EMBEDDINGS`-equivalent enum, git-repo provenance plumbing, and the produce/consume loop described in the exploration thread. Deferred: CLAUDE.md already flags this as on-demand work, gated on a downstream consumer asking for it.

(1) is cheap, correct for the common case, and doesn't block (4) — the skip condition just becomes stricter later ("output fresh AND manifest says unchanged").

## Known gaps (on purpose)

- **Re-uploads with identical bytes still re-embed.** S3 bumps `LastModified` whenever an object is replaced, even byte-for-byte. Acceptable: re-uploading a shard you don't expect to have changed is not a common path, and the cost is one extra embed, not a correctness issue.
- **Clock skew is not a concern.** Both timestamps are set by the same S3 cluster, not by client clocks.
- **No detection if an input is replaced *between* our HEAD and our read.** Race window of seconds; we'd embed the new content against the old timestamp. Not worth guarding here — the next run will catch it because output's `LastModified` is set to our upload time, which is earlier than the next re-upload.
- **Still no per-run provenance.** The output file contains no record of which input-`LastModified` it was produced from. Versioning is the mechanism for that; not in scope.

## Implementation shape

`io.py`:
- `InputKey` gains `last_modified: datetime | None = None` (default None keeps manually-constructed test fixtures valid).
- `list_input_keys` reads `obj["LastModified"]` from the `list_objects_v2` response — no extra HEAD.
- New `head_last_modified(bucket, key) -> datetime | None`: 404/NoSuchKey → None, other `ClientError` re-raised. Mirrors `object_exists` error semantics.

`pipeline.py`:
- `process_file` skip branch: `--force` → run; `head_last_modified(output)` is None → run; if `input.last_modified` is None (test fallback) → preserve old "output exists ⇒ skip" behavior; else compare. Two distinct log lines: "skip s3://… (output up-to-date)" and "reprocess s3://… (input newer than output: input=… output=…)".
- Same logic in the `dry_run` branch of `process_provider`.

Tests:
- `tests/test_pipeline.py`: replace `object_exists` patches with `head_last_modified`; add cases for output-newer → skip, output-older → reprocess, force → always reprocess.
- `tests/test_io.py`: test for `head_last_modified` (200, 404, other-error).
- `tests/test_e2e.py`: the existing dry-run case patches `object_exists` for "EXP-1910 already embedded"; update to patch `head_last_modified` returning an old-enough timestamp.

## Follow-ups

- When `impresso_essentials.versioning` integration lands (step not yet planned), the per-file `last_modified` check stays as the cheap first-pass filter; the manifest diff refines it.
