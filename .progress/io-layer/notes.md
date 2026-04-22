# I/O layer — decisions

## What we reuse from `impresso_essentials.io.s3`

| Use | Function | Why |
|---|---|---|
| S3 client/resource construction | `get_s3_client`, `get_s3_resource`, `get_bucket` | Already encapsulates host-URL and credential handling consistent with the rest of the Impresso stack. |
| Storage options for fsspec-compatible libs | `get_storage_options` | Matches what the rest of Impresso uses. |
| Upload (end-of-step, small hot path) | `upload_to_s3` | Correct behaviour, adequate for a single-shot upload per output file. We raise on `False` return to avoid silent drops. |

## What we intentionally do NOT reuse

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

### `upload_to_s3` — swallows exceptions, returns `bool`

We still call it, but we check the return value and raise. A silent failure would leave us believing an output was persisted when it wasn't.

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

## Existence check

`object_exists(bucket, key)` uses the boto3 client's `head_object` and treats `404` as False, any other `ClientError` as a real error. This backs the idempotent skip-if-output-exists behaviour.

## Not in scope for step 2

- Streaming **writes** to S3 (multipart upload) — local tempfile + single upload is simpler and correct. Reconsider only if we hit a file so large that local disk is a problem; unlikely.
- Dask/multiprocessing — `impresso_essentials.io.s3.fetch_files` returns dask bags; we're single-GPU/single-process by design.
