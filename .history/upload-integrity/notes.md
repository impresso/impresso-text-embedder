# Upload integrity — fix `MissingContentLength` and verify post-upload

## Context

A live Run:AI pod (A100, against Ceph RadosGW at Switch Engines) crashed on the first
output upload with `botocore.exceptions.ClientError: MissingContentLength`. Traceback:
`cli/create.py → pipeline.process_provider → io.upload_local_file → s3r.Bucket.upload_file`
→ `s3transfer` → `client.put_object`.

Root cause: since `boto3 1.36.0` (Jan 2025), S3 clients default to
`request_checksum_calculation="when_supported"`, which makes `PutObject` send the body as
`aws-chunked` with a trailer and no `Content-Length`. Ceph RadosGW rejects that.

The NGC 25.03 image we build on already ships a boto3 satisfying our old `>=1.34`
floor, and `Dockerfile` does `pip install --no-cache-dir .` into the NGC system
interpreter (no venv), so the image's boto3 — ≥1.36, not our lockfile's 1.35.21 — is
what ran. Confirmed by the traceback path `/usr/local/lib/python3.12/dist-packages/`.

## Fix

Two-part change in `src/impresso_text_embedder/io.py`:

1. **Disable flex-checksums.** New module-level
   `Config(request_checksum_calculation="when_required", response_checksum_validation="when_required")`
   passed to both `get_s3_client` and `get_s3_resource`. Restores a plain
   `Content-Length`'d PUT that Ceph accepts. Needs boto3 ≥1.36.5 / s3transfer ≥0.11.2
   so the setting propagates through the high-level `upload_file` — fixed in
   [boto/s3transfer#327](https://github.com/boto/s3transfer/issues/327). `pyproject.toml`
   bumps the floor accordingly; s3transfer is pulled in transitively and does not need
   an explicit pin.

2. **Verify each upload post-hoc.** `upload_local_file` now:
   - streams an MD5 of the local file (`hashlib.md5(usedforsecurity=False)`, 8 MB
     blocks) before the upload;
   - uploads as before;
   - issues one `HEAD` and compares `ContentLength` against the local size; for
     single-part ETags (no `-N` suffix) it also compares the returned ETag against
     the local MD5;
   - on any mismatch, best-effort-deletes the bad object (so the next run's
     `--skip-if-s3-exists` doesn't permanently hide the failure) and raises
     `RuntimeError`.

   One `INFO` line per successful upload records size, ETag, and whether multipart
   was used.

## Why these choices

- **`when_required` unconditionally, not env-gated.** Ceph is our only prod target;
  AWS S3 is still happy with the setting — it just opts out of the newly-added
  SDK-side checksum header, which servers don't require unless the operation does.

- **HEAD + ETag, not `ExtraArgs={"ContentMD5": ...}`.** `ContentMD5` was removed from
  `boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS` when boto3 moved to
  flex-checksums (current list has `ChecksumCRC32`, `ChecksumMD5`, `ChecksumSHA256`,
  etc., but no `ContentMD5`). Reaching `ContentMD5` would mean dropping to
  `client.put_object` directly and losing the TransferManager multipart path — a
  second code path that isn't worth the cost when a single `HEAD` plus a local MD5
  gives equivalent end-to-end coverage.

- **No `ExtraArgs={"ChecksumAlgorithm": "SHA256"}`.** That re-triggers `aws-chunked`
  framing and reintroduces the Ceph error.

- **Multipart ETag reconstruction is deferred.** S3's multipart ETag is
  `md5(concat(md5_of_each_part)) + "-" + N`, so reconstructing it requires knowing
  the exact per-part sizes the TransferManager actually used — an implementation
  detail that can drift silently. For our 50–500 MB shards, truncation or bit-flip
  would essentially always change either the size or at least one part's MD5 segment
  and therefore change the ETag shape; the size floor catches the realistic failure
  modes and the (always-on) single-part path gets bit-exact verification for free.

- **Delete-on-failure.** The alternative — leave the bad object — causes
  `--skip-if-s3-exists` to skip re-processing forever. A reproducible derivation
  (embeddings from text) has no loss from deletion. Two pods racing on the same
  key would already be a caller bug; the uploader is a single-slot
  `ThreadPoolExecutor(max_workers=1)` and keys are unique per
  `(provider, alias, year, model_slug)`.

- **`hashlib.md5(usedforsecurity=False)`.** Required on FIPS-enabled RHEL / NGC
  images — MD5 is disallowed by default; this kwarg is the documented escape hatch
  for non-security uses.

## Known gaps / revisit triggers

- Multipart bit-exact verification. Would need either (a) a pinned upload
  `TransferConfig` whose chunk size matches what we assume at verify time, or (b)
  dropping to `put_object` (single-part only; caps us at 5 GB per shard). Neither
  is worth it until we see a failure the size-floor misses.
- SSE-KMS or SSE-C on the output bucket. Would break the ETag ≡ MD5 identity; we'd
  need to fall back to size-only or adopt flex-checksums on an AWS-only path. Not
  currently used on Switch Engines.
- Byte-identical re-uploads that differ only in header metadata. Out of scope here;
  the orthogonal `re-embed on input change` logic already covers input-side drift.

## Upstream

`impresso/impresso-essentials/io/s3.py` does not address this: it passes no `Config`
to boto3 and `upload_to_s3` swallows failures as `return False`. Our fix here is
strictly stronger (raises + verifies) and a candidate upstream contribution once
proven in this pipeline.
