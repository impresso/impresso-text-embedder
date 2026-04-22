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

## Skip-if-exists

Default on (matches old `--quit-if-s3-output-exists`). CLI flag `--force` disables it. Check is one `head_object` per input file at the top of `process_file`.

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
    [--include-text]                          # default: False (text level only)
    [--alias <code> ...]                      # filter
    [--year-min <int>] [--year-max <int>]
    [--force]                                 # overwrite existing outputs
    [--dry-run]                               # list files + skip/process counts, no work
    [--log-level INFO|DEBUG|WARNING|ERROR]
```

`<model-slug>` for the output path strips the `Alibaba-NLP/` vendor prefix: `Alibaba-NLP/gte-multilingual-base → gte-multilingual-base`. Revision is **not** appended to the slug in this pass — weights at different revisions go to the same output path. If that turns out to be a problem, we'll introduce `--model-slug` as an explicit override.

## Deferred (intentionally)

- **Async prefetch + async upload.** A thread pool sized (1 prefetcher, 1 uploader) would let file N+1 download while file N encodes. Worth doing only after measurement confirms GPU idle time between files. Adds complexity (lifecycle, back-pressure, error handling); don't pay that before proving it matters.
- **Multi-GPU / DDP.** Explicit non-goal per `CLAUDE.md`.
- **Resumability mid-file.** A file either completes or is redone. Explicit non-goal per `CLAUDE.md`.
- **Dry-run that renders the output.** `--dry-run` here just lists what would be processed/skipped; it doesn't run the model.
