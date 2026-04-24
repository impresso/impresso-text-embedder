# I/O layer — decisions

## Why we dropped `impresso-essentials` entirely (step 12)

Originally we imported `get_s3_client`, `get_s3_resource`, and `upload_to_s3`
from `impresso_essentials.io.s3`. Building on `nvcr.io/nvidia/pytorch:25.03-py3`
made that untenable:

1. **Hardcoded pins in the package metadata.** `impresso-essentials==1.4.1`
   declares `numpy==2.2.1`, `dask>=2024.9.0`, `pandas>=2.2.2`, `pyarrow>=17.0.0`,
   `nltk==3.9.1`, and ~20 more. `pip install .` on the NGC image tries to
   replace NGC's `numpy==1.26.4` with numpy 2.2.1, which breaks the apex /
   NCCL / transformer-engine / xformers wheels (all compiled against the
   numpy-1.x C-ABI).
2. **`--no-deps` doesn't rescue us.** `impresso_essentials/io/s3.py` does
   `import dask.bag as db` at module level. The module can't be imported
   without at least `dask`, and `dask` pulls in pandas/numpy through its
   own top-level imports (`toolz`, `cloudpickle`, `partd`) — so the `--no-deps`
   cascade would have to include every one of those, defeating the point.
   Notably, **none of the three helpers we use actually calls dask** — `db` is
   only referenced by `read_s3_issues` and `fetch_files`, which we never call.
3. **The helpers are trivial.** `get_s3_client` and `get_s3_resource` are
   ~15 lines each of `boto3.{client,resource}("s3", ...)` driven by
   `SE_ACCESS_KEY`/`SE_SECRET_KEY`/`SE_HOST_URL` env vars. `upload_to_s3`
   is a 5-line `bucket.upload_file(...)`. Total: ~40 lines.

**Decision:** vendor the three helpers into `src/impresso_text_embedder/io.py`,
drop `impresso-essentials` from `pyproject.toml`, and don't install it in the
Docker image. A build-time `assert numpy.__version__.startswith('1.26')` in the
Dockerfile guards against any silent numpy upgrade from another source.

If a future need surfaces for something else from `impresso-essentials`
(e.g. the `versioning` manifest system), prefer a **minimal vendored snippet**
or push upstream for a runtime-slim extras set — do **not** re-introduce the
full package against the NGC image.

## What we intentionally do NOT reuse (even before vendoring)

### `read_jsonlines` — fully loads the file into memory

Source (paraphrased from the installed package):

```python
body = s3r.Object(bucket_name, key_name).get()["Body"]
data = body.read()                            # full download
text = bz2.decompress(data).decode("utf-8")   # full decompress
for line in text.split("\n"):                 # in-memory split
    if line != "":
        yield line
```

The yearly `.jsonl.bz2` shards can be tens of MB compressed / hundreds of MB uncompressed per file. Loading each one fully before encoding:
- ties peak RSS to worst-case file size;
- adds an unbounded stall between file boundaries while the GPU sits idle, which contradicts the "GPU must be the bottleneck" requirement in `CLAUDE.md`;
- blocks the prefetching strategy planned for step 6.

Instead: open the S3 body stream, wrap in `bz2.open` (which can decompress chunk-by-chunk), and iterate lines. This keeps memory bounded and lets a prefetcher run file N+1 in parallel with the encode of file N.

### `upload_to_s3` (upstream) — swallows exceptions, returns `bool`

Our vendored `upload_local_file` raises on failure. A silent failure would leave us believing an output was persisted when it wasn't.

### `list_providers_and_aliases` — returns providers and aliases, not per-year keys

We need yearly file keys under a given provider; this helper stops at the alias level. For listing, we paginate directly via `list_objects_v2` on the S3 client and filter by `.jsonl.bz2` suffix. Cheaper than a glob for this exact shape.

## Input/output key convention

From `CLAUDE.md`:

```
input:  s3://<input-bucket>/[<input-prefix>/]<provider>/<alias>/<alias>-<year>.jsonl.bz2
output: s3://<output-bucket>/embeddings/docs/<model-slug>/<provider>/<alias>/<alias>-<year>.jsonl.bz2
```

Helpers:
- `build_output_key(provider, alias, year, model_slug) -> str`
- `parse_input_key(key, input_prefix="") -> (provider, alias, year)` — the reverse

Filename discipline: `<alias>-<year>.jsonl.bz2`. If real inputs ever violate this, parsing fails loudly (assertion) rather than producing a mangled output path.

## Output freshness check

`head_last_modified(bucket, key)` uses the boto3 client's `head_object` and returns the object's `LastModified` timestamp (tz-aware UTC), or `None` on `404`/`NoSuchKey`/`NotFound`. Any other `ClientError` is re-raised. Used by the pipeline to compare output vs input timestamps — see `.progress/reembed-on-change/notes.md`.

## Not in scope for step 2

- Streaming **writes** to S3 (multipart upload) — local tempfile + single upload is simpler and correct. Reconsider only if we hit a file so large that local disk is a problem; unlikely.
- Dask/multiprocessing — `impresso_essentials.io.s3.fetch_files` returns dask bags; we're single-GPU/single-process by design.
