# `impresso-embed-create` — design notes

## Scope of step 6

Ship a correct, single-threaded, per-file orchestrator end-to-end. Async prefetch/upload overlap is **deferred** until we can measure idle-GPU windows on a real A100 — see "Deferred" below.

## Module split

| File | Role |
|---|---|
| `embed.py` | Per-record embedding: given a parsed input record + model + level, produce the output schema object(s). Batching for text-level happens here via an explicit buffer/flush helper. |
| `pipeline.py` | Per-file orchestration: stream records from input, route to `embed.py`, write the output `.jsonl.bz2`, upload. |
| `cli/create.py` | Arg parsing, env/logging setup, provider-level loop over files. |

Keeping `embed.py` separate means the per-record logic can be exercised in unit tests without touching S3 or disk.

## Text-level batching

Old code encoded one record at a time with `batch_size=8`. That massively underutilizes an A100 — a single 300M-param forward on a few hundred tokens is dominated by launch overhead. For text-level we:
- accumulate records into a buffer of up to `--batch-size`,
- encode the whole buffer in one `encode_texts` call,
- emit one JSON line per record.

For sentence and chunk levels the old per-record approach is fine — each call already batches many items (sentences / chunks) inside a single record.

## Output file

Written locally first as a `.jsonl.bz2` file (`bz2.open(..., "wt", encoding="utf-8")`), then uploaded. Local-tempfile + single upload is simpler than streaming-multipart and correct for the file sizes we handle.

## Output schema (text level)

Source of truth: Impresso document-embeddings JSON schema (`embeddings-docs.schema.json`, imported in the `impresso-schemas` repo as `Document Embeddings JSON Schema`).

| field       | required | type    | notes                                                                |
|-------------|----------|---------|----------------------------------------------------------------------|
| `ci_id`     | yes      | string  | canonical content-item id (e.g. `actionfem-1940-01-08-a-i0001`)      |
| `model_id`  | yes      | string  | id of the model that produced the embedding (see below)              |
| `embedding` | yes      | array   | list of floats, rounded to 5 decimals in `schema.TextRecord.to_dict` |
| `size`      | yes      | integer | embedding dimension = `len(embedding)`                               |
| `ts`        | no       | string  | RFC3339 UTC, e.g. `2024-10-09T09:29:02Z`                             |
| `ci_type`   | no       | string  | echo of the input record's `tp` (e.g. `ar`, `page`)                  |

The writer lives at `schema.TextRecord`; the validator's required-key list (`validate._check_text_record`) mirrors the first four rows, plus `size == len(embedding)`.

### Divergences fixed

An earlier version of the writer emitted `{id, ts, embedder, len, embedding, text?}`:
- `id` → renamed to `ci_id` (matches schema + matches what the old `main`-branch pipeline wrote).
- `embedder` → renamed to `model_id` (schema-compliant key).
- `len` (char length) → dropped (not in the schema).
- `text` (and the `--include-text` CLI flag that gated it) → dropped. Debug-inspection of the source text belongs next to the input file, not in the embedding file.
- `size` → added (required by the schema). Used by the validator to cross-check `len(embedding)`.

Nice side effect: files produced by the old pipeline (e.g. `s3://141-processed-data-staging/embeddings/docs/embeddings_gte_v1-1-0/...`) are already schema-compliant; after this refactor they validate structurally without changes.

### Deferred — model_id value format

The schema says `model_id` must follow the Impresso slides convention (e.g. `doc-embeddings_gte-multilingual-v1.0.0`). Today the writer passes `build_embedder_tag(model_name, model_revision)` unchanged, which — since the revision pin landed (step 14, `.progress/model-revision-pin/notes.md`) — produces `Alibaba-NLP/gte-multilingual-base@f7d567e` by default (and `@default` only when a caller explicitly passes `model_revision=None`, e.g. some unit tests). Field name is schema-compliant; the *value* string is not yet. Aligning the value is a separate follow-up — needs the current Impresso model-id spec pinned.

Cross-reference: `CLAUDE.md` → "Decisions recorded" has the one-line entry.

## Skip-if-exists

Default on (matches old `--quit-if-s3-output-exists`). CLI flag `--force` disables it. Check is one `head_object` per input file at the top of `process_file`.

## Sampling / smoke tests

`--limit N` truncates the S3 listing after N keys (lexicographic order of `list_objects_v2` — same order the pipeline already walks). Meant for quick end-to-end checks against a new image, GPU node, or batch-size setting without hand-rolling `--alias` / `--year-min` / `--year-max` filters that match the layout exactly.

Implemented by wrapping the `list_input_keys(...)` iterator with `itertools.islice(listing, limit)` inside both `pipeline._plan_files` and `pipeline._dry_run_summary`. Listing-only filters (`--alias`, `--year-*`) run first (inside `list_input_keys`), so `--limit` slices the already-filtered stream.

Semantic: **applied pre-skip**. An already-processed shard (output fresher than input) still counts toward the limit and shows up as `skipped` in the summary. This keeps behaviour predictable and avoids paginating the entire bucket to find N "new" files — combine with `--force` if you want N *re-runs* rather than N "scan slots". `--limit 0` is accepted and processes zero files; negative values are rejected at parse time with a clear error. No interaction with the prefetch/upload overlap — the pipeline walks a truncated list, that's all.

## Error policy

- Malformed JSON line → log and skip that line, keep going.
- Malformed token offsets / missing fields → log and skip record.
- Failed upload → raise (never silent).
- Upstream S3 read error → raise (fail the file; another run will retry).

## CLI flags

Minimal first pass — add more when a concrete need appears.

```
impresso-embed-create \
    --provider <CODE>                         # required
    --input-bucket <name>                     # required
    --output-bucket <name>                    # required
    [--input-prefix <p>]                      # default: ""
    [--model-name Alibaba-NLP/gte-multilingual-base]
    [--model-revision <ref>]                  # default: None
    [--embedding-level text|sentence|chunk]   # default: text
    [--chunking-strategy semantic]            # default: semantic (only used if level=chunk)
    [--batch-size N]                          # default: 64 for text, caller-visible
    [--min-char-length 400]
    [--content-type ar|page ...]              # default: ar
    [--normalize-embeddings]                  # default: False
    [--alias <code> ...]                      # filter
    [--year-min <int>] [--year-max <int>]
    [--force]                                 # overwrite existing outputs
    [--dry-run]                               # list files + skip/process counts, no work
    [--limit N]                               # process at most the first N listed shards
    [--log-level INFO|DEBUG|WARNING|ERROR]
```

`<model-slug>` for the output path strips the `Alibaba-NLP/` vendor prefix: `Alibaba-NLP/gte-multilingual-base → gte-multilingual-base`. Revision is **not** appended to the slug in this pass — weights at different revisions go to the same output path. If that turns out to be a problem, we'll introduce `--model-slug` as an explicit override.

## Deferred (intentionally)

- **Async prefetch + async upload.** A thread pool sized (1 prefetcher, 1 uploader) would let file N+1 download while file N encodes. Worth doing only after measurement confirms GPU idle time between files. Adds complexity (lifecycle, back-pressure, error handling); don't pay that before proving it matters.
- **Multi-GPU / DDP.** Explicit non-goal per `CLAUDE.md`.
- **Resumability mid-file.** A file either completes or is redone. Explicit non-goal per `CLAUDE.md`.
- **Dry-run that renders the output.** `--dry-run` here just lists what would be processed/skipped; it doesn't run the model.
