# Model-revision pin — design notes

## Why pin

`sentence-transformers` resolves the HF Hub ref `main` at load time. If
upstream re-tags `Alibaba-NLP/gte-multilingual-base` (tokenizer tweak,
re-quantized weights, anything), two otherwise-identical runs produce
different vectors. That silently:

- breaks bit-reproducibility of golden files used by
  `impresso-embed-validate`;
- violates the `--tol 1e-4` cosine contract from
  `.progress/validation-metric/notes.md`;
- makes the A100↔H100 drift experiment (still open in `CLAUDE.md` →
  "Still open — needs real GPU time") un-interpretable, because the
  variable under test is supposed to be the GPU arch, not the weights.

Pinning to a specific commit SHA takes that variable off the table.

## What was pinned

- Model: `Alibaba-NLP/gte-multilingual-base`
- Revision: `f7d567e`

## Where the pin lives

Single source of truth: `src/impresso_text_embedder/model.py`:

```python
DEFAULT_MODEL_NAME     = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_MODEL_REVISION = "f7d567e"
```

`load_model(revision=DEFAULT_MODEL_REVISION, …)` uses it as the default.
`cli/create.py` imports both constants and threads them as argparse
defaults for `--model-name` / `--model-revision`.

The Makefile mirrors the pin for auditability of production jobs:

```make
CREATOR_NAME     ?= Alibaba-NLP
HF_MODEL_NAME    ?= gte-multilingual-base
HF_MODEL_VERSION ?= f7d567e
```

and `runai-submit` forwards them as explicit
`--model-name $(CREATOR_NAME)/$(HF_MODEL_NAME) --model-revision $(HF_MODEL_VERSION)`
on the CLI. The explicit pass-through means the submitted Run:AI command
line carries the pin (visible in `runai describe job`), not just an
implicit Python default — so the pin is recoverable from cluster logs
even if the image changes.

`.env.docker.example` documents the overrides as commented-out lines
(the `?=` in the Makefile honours `.env.docker` overrides via
`-include .env.docker`).

## How to bump

Two separate motions on purpose — the choice signals intent:

- **One-off experiment (e.g. evaluating a candidate revision on a
  specific provider):** set `HF_MODEL_VERSION=…` in `.env.docker` and
  re-run `make runai-submit`. No code change, no rebuild. The Python
  default stays on the current pin — tests and local runs are
  unaffected.
- **Permanent bump of the shipped model:** edit
  `DEFAULT_MODEL_REVISION` in `model.py` *and* `HF_MODEL_VERSION` in the
  Makefile to keep them in sync. Update the assertion in
  `tests/test_cli_create.py::test_parser_pins_model_revision_by_default`
  and the two `@<revision>` strings in `tests/test_cli_create.py` +
  `tests/test_e2e.py`. Add a dated line to this file recording the new
  SHA and why (new weights, tokenizer fix, whatever). Consider bumping
  the Impresso model slug (`pipeline.MODEL_SLUG_OVERRIDES`) too if the
  change is visible enough that you want to fork the output path.

## What did **not** change

- `pipeline.MODEL_SLUG_OVERRIDES` — the output path still resolves to
  `embeddings/docs/embeddings_gte_v1-1-0/…` regardless of revision.
  That matches the "Decisions recorded" note in `CLAUDE.md`: different
  revisions of the same Impresso-blessed model share one output path
  because the versioned slug is curated, not auto-derived from the SHA.
- `build_embedder_tag` — still renders `name@revision`, and falls back
  to `@default` only when a caller passes `revision=None` explicitly
  (e.g. `tests/test_pipeline.py::_cfg`). Once the CLI default is
  `"f7d567e"`, production writes carry `@f7d567e` in every output
  record's `model_id`.

## Verification

- `uv run impresso-embed-create --help` shows
  `--model-revision f7d567e` as the default.
- `make -n runai-submit PROVIDER=X INPUT_BUCKET=i OUTPUT_BUCKET=o`
  dry-prints a command whose trailer contains
  `--model-name Alibaba-NLP/gte-multilingual-base --model-revision f7d567e`.
- `uv run pytest -x` stays green (three test strings were updated for
  the new default — see the step-4 bullet in the plan file and the
  step-14 entry in `.progress/plan.md`).

## Known gaps

- Pinning the HF revision does not replace the still-deferred
  `impresso_essentials.versioning` manifest — that remains on the
  "add when a consumer needs it" list in `CLAUDE.md`.
- We do not verify at load time that the resolved commit matches the
  requested SHA (sentence-transformers raises on an outright-missing
  revision, but a silently-redirected ref would slip through). An
  optional guard call to `huggingface_hub.HfApi().model_info(repo_id,
  revision=revision).sha` with an assertion is a cheap follow-up if we
  see evidence of drift.
