# `--normalize-embeddings` removal — design notes

## What was there

- CLI flag `--normalize-embeddings` (`store_true`, CLI default `False`).
- `EncoderConfig.normalize_embeddings: bool = True` (dataclass default `True` —
  disagreed with CLI default).
- Threaded through three call sites into `model.encode(normalize_embeddings=…)`,
  which requests an extra L2 pass at the end of the encode pipeline.

## Why it was dead

`Alibaba-NLP/gte-multilingual-base` (the only blessed embedder) ships
`modules.json = [Transformer, Pooling, Normalize]`. Every vector that comes
out of `model.encode(...)` is already L2-unit. An additional L2 pass on a
unit vector is the identity. So for the only model we run, the flag was a
no-op regardless of value. The mismatched defaults (CLI `False` vs
EncoderConfig `True`) were invisible on this model but would produce
different behaviour the moment we swapped to a non-normalising encoder —
exactly the wrong way for a foot-gun to surface.

## Decision

1. Delete the CLI flag and the `EncoderConfig` field.
2. Hardcode `model.encode(normalize_embeddings=False)` in `encode_texts` —
   we trust the model's built-in `Normalize` module and don't pay for a
   redundant L2.
3. Add a startup assertion in `load_model`: the last `SentenceTransformer`
   module (it's an `nn.Sequential`, so `model[-1]`) must be an instance of
   `sentence_transformers.models.Normalize`. Otherwise raise `RuntimeError`
   naming the model `name@revision`.
4. `MeanPoolStrategy`'s post-mean L2 renorm is **untouched** — it operates
   on the *aggregated* mean vector, which is not unit-norm even when its
   input chunks are. Different invariant, different stage.

The cosine validation contract (`--tol 1e-4`) is now a load-time invariant
instead of a runtime hope.

## Why not auto-detect (rejected alternative)

Auto-detect would read `modules.json` (or check `model[-1]`) at load time
and *adapt* the per-chunk L2 behaviour: leave it off if `Normalize` is
present, turn it on otherwise. Considered and rejected:

- **Hides aggregator-relevant semantics inside the encoder layer.** Whether
  chunks reach MeanPool unit-norm is a property the aggregator cares about
  (it determines whether the mean is direction-only or implicitly
  norm-weighted). Auto-detect would couple that property to whatever
  module list the model happens to ship — a reader of `encode_texts(...)`
  would have to trace into `load_model` to know what's actually happening.
- **Final against a model swap, not against an aggregator swap.** The
  unit-chunk assumption is currently load-bearing for `MeanPoolStrategy`
  but not for, say, a future `first-chunk` or attention-weighted aggregator.
  Auto-detect doesn't help with the latter and just delays the
  disentangling.
- **YAGNI.** No encoder swap is on the roadmap. The model is revision-pinned
  (`f7d567e`), the slug is registered, `transformers<5` is pinned around
  this exact model, and the validation contract is defined for it. Paying
  complexity now for an option we may never exercise violates CLAUDE.md's
  "don't design for hypothetical future requirements."

## Forward-compat — what to do on a model swap

If a future encoder swap *does* drop the `Normalize` module, the assertion
in `load_model` will fire on startup. Don't re-add the flag. The right
response is to make the unit-chunk assumption explicit at the aggregator
layer: either

- enforce it inside `MeanPoolStrategy.aggregate` (assert `‖v_i‖ ≈ 1` for
  all chunks, and/or normalise them on entry), or
- expose it as a per-aggregator contract (e.g. an `expects_unit_inputs`
  property on `AggregationStrategy`) that the encoder honours by inserting
  a normalisation step when it would be violated.

Either way, the change is a deliberate code review with a test, not a
silent CLI default.

## Files touched

- `src/impresso_text_embedder/model.py` — assertion + hardcoded
  `normalize_embeddings=False` in `encode_texts`.
- `src/impresso_text_embedder/embed.py` — dropped `EncoderConfig.normalize_embeddings`
  + 3 call-site args.
- `src/impresso_text_embedder/cli/create.py` — dropped argparser block + dropped
  `EncoderConfig(...)` kwarg.
- `tests/test_model.py` — updated mocks to satisfy the assertion; added a test
  that asserts `RuntimeError` when `model[-1]` is not a `Normalize`.
- `tests/test_cli_create.py`, `tests/test_pipeline.py`, `tests/test_embed.py` —
  removed flag/field references.
- `README.md`, `CLAUDE.md`, `.progress/long-doc-chunking/notes.md` — trimmed
  flag mentions; added decision entry pointing here.
