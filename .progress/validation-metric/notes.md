# `impresso-embed-validate` — metric and tolerance

## Purpose

Two modes in one CLI:

1. **Structural validation** (no `--target`): prove a freshly produced `.jsonl.bz2` is well-formed — all lines parse, required fields present, all embeddings same length, no NaN/Inf, timestamps in the expected format.
2. **Comparison** (`--target <path>`): prove the produced file matches a reference embedding file per-record within tolerance. Used for regression checking when we upgrade PyTorch, CUDA, flash-attn, chunker, etc.

## Metric: cosine distance on L2-normalized vectors

`distance(a, b) = 1 - (â · b̂)` where `â = a / ||a||` (eps-guarded).

Reasons:
- `gte-multilingual-base` is trained for cosine similarity; this is the natural distance for these embeddings.
- Independent of whether the user passed `--normalize-embeddings` on both sides (we normalize here regardless), so the comparison is invariant to scale.
- Bounded in `[0, 2]`, so a single absolute tolerance is interpretable.
- Robust under small rounding / bf16-vs-fp32 drift: sign-preserving, magnitude-stable.

Rejected alternatives:
- **L∞ on raw vectors**: brittle under bf16 rounding; a single large-magnitude component can blow up while the angle is essentially unchanged.
- **L2**: dominated by whichever axis happens to have the largest absolute value; less interpretable across record lengths.

## Default tolerance: `1e-4`

- Bit-identical outputs score `0` trivially.
- Rounding to 5 decimals on disk (our schema) caps per-component error at `5e-6`; cumulative effect on cosine distance for ~1000-D vectors is comfortably below `1e-4`.
- Real bf16-vs-fp32 drift between hardware has been observed in practice around `1e-4`–`1e-3`; the default allows the former and flags the latter.
- Tightenable via `--tol` when you want to detect any change at all (set to `0`) or loosen for cross-hardware runs.

## Matching records

- Text level: by top-level `id`.
- Sentence level: by `(ci_id, sent_id)`.
- Chunk level: by `(ci_id, chunk_id)`.

If a record exists in one file but not the other, it's reported as a mismatch, not just a warning. The validate CLI exits non-zero in that case.

## Exit codes

- `0` — clean.
- `1` — validation failed (structural error, missing records, or distance > tol).
- `2` — CLI/config error (argparse handles this automatically).

## Out of scope

- **Fuzzy id matching.** Ids must match exactly.
- **Re-ordering tolerance.** We build by-id maps on both sides before comparing, so record *order* in the two files is irrelevant. But items inside a record (sentences, chunks) use the sent_id / chunk_id as the match key.
- **Text content equality.** We only compare vectors; even if `include_text` is set on both sides, we don't diff the text.
- **Cross-level comparison.** Comparing a text-level output to a sentence-level output is meaningless and unsupported.
