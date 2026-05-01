# corpus-fetch — design notes

Materialise the per-language manifest produced by step 1 into a single
`.jsonl.bz2` corpus shard that downstream chunking-strategy sweeps consume,
and upload it to a sandbox S3 location. This is the "fetch the rebuilt text"
step — it does not embed, chunk, or evaluate; it just resolves manifest
ci_ids to their full text payload.

## Inputs / outputs

- **Input**: `tmp/chunking-eval/corpus-manifest.jsonl` (400 entries from step 1).
- **Output**:
  - Local: `tmp/chunking-eval/corpus-v1.jsonl.bz2` (kept for debugging; the
    `--local-output` flag controls whether the local copy survives the run).
  - S3: `s3://140-processed-data-sandbox/chunking-eval/corpus/corpus-v1.jsonl.bz2`.
  - Per-record output schema: `{ci_id, lg, year, provider, alias, len_chars,
    ocrqa, tp, ft, sents, lingproc_path?}`. Manifest fields carry through;
    `ft` is the rebuilt full text (precomputed on the rebuilt record where
    present, else reconstructed from token offsets via the production
    `text.rebuild_ft_from_offsets` helper); `sents` is preserved for any
    downstream sentence-aware chunker that wants to use the upstream
    tokenisation rather than re-splitting.

## Mechanism

1. **Read** the manifest into a list of `ManifestEntry` rows (keeps original
   order).
2. **Group** by `(rebuilt_bucket, rebuilt_key)` → `{ci_id: entry}`. The
   manifest's 400 entries collapse into 222 unique rebuilt files; median
   per-file count is 1, max 9. Grouping avoids re-fetching the same yearly
   shard for the second/third manifest entry that lives in it.
3. **Stream** each rebuilt shard via :func:`io.iter_jsonl_bz2` (a single
   ``Object.get()['Body']`` read piped through ``bz2.open`` in text mode),
   picking out only the manifest's ci_ids and breaking early when every
   wanted ci_id has been picked up. With the manifest's median 1
   record-per-shard, the early-break means we read only the prefix
   needed — the rest of the bz2 body is never decompressed nor
   transferred.
5. **Build** an output record per matched ci_id: manifest metadata + the
   rebuilt record's `tp`/`ft`/`sents`. `ft` falls back to
   `rebuild_ft_from_offsets(sents)` when missing — same byte-for-byte
   behaviour as the production pipeline's input handling.
6. **Workers**: a `ThreadPoolExecutor` (default 4) submits per-file fetches
   in parallel. Output is assembled in **manifest order** by indexing into
   the fetched dict at the end, so completion order does not affect the
   shard's record order — the downstream sweep sees deterministic input.
7. **Upload** the local shard to `s3://140-processed-data-sandbox/...` via
   :func:`io.upload_local_file`, which carries the production integrity
   checks (size match + single-part ETag/MD5) and Ceph-RadosGW-compatible
   checksum config.

## Worker-count calibration

The script defaults to `--max-workers 16`. Each worker holds **one**
streaming connection (single-part `Object.get()`) so 16 workers ≈ 16
concurrent S3 connections — well within Ceph RadosGW's comfort zone and
small enough that aggregate egress dominates rather than connection
overhead. Calibrated empirically on the 222-file manifest after rejecting
the multipart-parallel-download alternative (see "Rejected alternatives"
below); revise if the file mix changes shape.

## S3 destination convention

Research outputs go under `s3://140-processed-data-sandbox/chunking-eval/...`
**not** under the production `embeddings/docs/<model-slug>/...` path. This
is a hard rule recorded in `CLAUDE.md` ("Research outputs … live under
`tmp/` or `.out/` locally — they do **not** write to the production output
prefix"). The same bucket is used to keep RCP-side credentials simple, but
the prefix segregates research artefacts from production embedding shards
unambiguously.

The `corpus-v1` suffix gives us a clear bump path: regenerating the
manifest or schema warrants a new `corpus-vN.jsonl.bz2` rather than
overwriting the v1. `--output-key` is parameterised so the bump is a
flag-flip, not a code change.

## Rejected alternatives

- **Multipart-parallel `download_to_local`** (the production pipeline's
  default for embedding files). Tried this after the first streaming
  pass *appeared* slow; turned out to be slower in practice because
  multipart insists on fetching the **entire** yearly shard before any
  record can be scanned, while streaming + early-break stops as soon as
  the wanted ci_ids are seen. With the manifest's median 1 record per
  yearly shard, the early-break wins by 5-10× even though each
  streaming connection is single-threaded. The production pipeline
  embeds *every* record per file so multipart is the right call there;
  for the long-doc-corpus assembly we want the opposite trade-off.
  Single-stream `iter_jsonl_bz2` × 16 outer workers is the chosen
  pattern.
- **S3 Select / range-scan to extract specific ci_ids** without downloading
  the whole shard. Rejected: Ceph RadosGW (Switch Engines) does not
  guarantee S3 Select, and the manifest's median 1-record-per-file means
  the speedup ceiling is bounded by how fast we can locate the record
  inside the shard — and the records aren't keyed by byte offset, so
  finding them still requires a sequential scan. Multipart-parallel
  download + local scan ends up at the same big-O with no extra
  infrastructure dependency.
- **Stream and write directly without an intermediate local file**.
  Rejected: would bypass `upload_local_file`'s integrity verification
  (size + ETag/MD5 check) which we want; it would also tie output ordering
  to fetch-completion ordering, defeating the manifest-order guarantee
  that lets the downstream sweep reproduce the same record sequence
  across runs.
- **Separate per-language output shards** (one `corpus-fr.jsonl.bz2` and
  one `corpus-de.jsonl.bz2`). Rejected: adds a join the downstream sweep
  doesn't need; the per-record `lg` field plus the manifest's stable
  ordering give per-language slicing for free without splitting the
  artefact.
- **Drop `sents` from the output to halve the file size**. Rejected:
  cheap to keep (bz2 compresses sentence offsets well — adds ~2× over
  ft-only based on the rebuilt record shape) and `sents` is the upstream
  authoritative tokenisation. The sentence-aware chunkers
  (`token-budget`, future semantic variants) currently re-split via a
  regex because the wiring to use the upstream `sents` is deferred (see
  `chunking/token_budget.py` docstring); preserving `sents` here keeps
  the door open.
- **Filter records by `tp == 'ar'`**. Rejected: the manifest is already
  filtered to articles by the corpus-selection step (which used the
  aggregator schema's `tp == 'article'`). Re-filtering at fetch time
  would be redundant and risks dropping records if the aggregator and
  rebuilt schemas disagree on the tag string. The output simply passes
  the rebuilt record's `tp` through unchanged.
- **`huggingface_hub` / `datasets` to materialise the corpus**. Rejected:
  we already have all the S3 plumbing in `io.py`; pulling in another
  ecosystem dependency for a one-off research artefact is not justified.

## Open items

- **chars_per_token calibration follow-up**. With the corpus shard
  materialised, the deferred chars_per_token-calibration deliverable
  (open item O5 in `.progress/plan.md`) becomes a one-liner: tokenise
  `ft` per record with the gte-multilingual-base tokenizer, group by
  `lg`, and emit empirical chars-per-token. Revise step 1's threshold
  if the gap is >10%.
- **Corpus-shard versioning policy**. Right now `corpus-v1` is a
  hand-bumped suffix. If the eval starts producing many variants
  (different `--n-per-lg`, different OCR cutoff, etc.) we may want a
  manifest-fingerprint-derived path. Defer until we actually have a
  second variant.
- **Coverage check**. The fetch logs every manifest ci_id that is not
  found in its rebuilt source file as a WARNING. Step 1's manifest
  pointed at files that exist — the expected number of misses is zero.
  If the run reports any, investigate whether the aggregator's
  `source_key` got out of sync with what `122-rebuilt-final` actually
  carries.

## Reproducing

```
uv run python -m impresso_text_embedder.research.corpus_fetch \
  --manifest    tmp/chunking-eval/corpus-manifest.jsonl \
  --local-output tmp/chunking-eval/corpus-v1.jsonl.bz2 \
  --output-bucket 140-processed-data-sandbox \
  --output-key   chunking-eval/corpus/corpus-v1.jsonl.bz2 \
  --max-workers 16
```

`--no-upload` skips the S3 upload (requires `--local-output`); pass
`--max-workers N` to retune for a different bandwidth profile.
