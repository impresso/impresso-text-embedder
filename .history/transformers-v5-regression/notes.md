# Transformers v5 regression on `Alibaba-NLP/new-impl` — pin & rationale

Status: **resolved** by pin. Revisit when either upstream ships a fix or we
replace `gte-multilingual-base` with a maintained model.

## Problem

On the Run:AI debug pod (`nvcr.io/nvidia/pytorch:25.03-py3`, A100), loading the
model via `sentence-transformers` and encoding a trivial 6-token input
crashed with a CUDA device-side assert:

```
/opt/pytorch/.../IndexKernel.cu:93: operator(): block: [0,0,0], thread: [0,0,0]
  Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

With `CUDA_LAUNCH_BLOCKING=1` the synchronous trace lands at:

```
File ".../transformers_modules/.../new_hyphen_impl/.../modeling.py", line 392, in forward
    rope_cos = rope_cos[position_ids].unsqueeze(2)
RuntimeError: CUDA error: device-side assert triggered
```

The repro is the Makefile's own smoke test — `m.encode('This is a test!')` —
so the input is ~6 tokens. Not a sequence-length bug.

## Red herrings we ruled out

Hours were spent on each of these before arriving at the real cause; noting
them so future me doesn't repeat the walk:

1. **Sequence length > 8192 (RoPE table overrun).** Initially plausible — BL
   content items are long, the model's `max_position_embeddings=8192`, the
   failing line is literally a RoPE lookup. Refuted by the 6-token repro.
2. **xformers' unpad path corrupting `position_ids`.** Repros without
   xformers (`IMPRESSO_DISABLE_XFORMERS=1` equivalent — unset the
   `config_kwargs`). Same crash, same line.
3. **BL-specific content pathology.** Refuted — BNL crashes too, and so does
   the trivial smoke test.
4. **NGC 25.03 / torch 2.7 nightly CUDA mismatch.** NGC 25.03 is fine;
   upstream issues explicitly name this same crash on release torch builds.
5. **Missing `max_seq_length` cap at load time.** We almost landed a
   `model.max_seq_length = min(...)` patch in `model.py::load_model` — would
   have been a placebo. With 6 tokens, there's nothing to cap.

## Root cause

`transformers>=5` rewrote `from_pretrained` to use **meta-device
materialization**: parameters and buffers are first allocated on the meta
device (no storage), then storage is allocated and initialized as layers
are loaded. A bug in that flow leaves buffers registered with
`persistent=False` pointing at allocated-but-uninitialized storage after
load.

Alibaba's custom modeling file (`Alibaba-NLP/new-impl@40ced75`, cached on
disk as `transformers_modules/.../new_hyphen_impl/40ced75.../modeling.py`)
registers its position index buffer with:

```python
self.register_buffer(
    "position_ids",
    torch.arange(config.max_position_embeddings).expand(1, -1),
    persistent=False,
)
```

After loading under transformers v5, `self.position_ids` contains uninitialized
memory — values like `94764670470272`. At forward time the modeling code
does `rope_cos[position_ids]`, garbage indices are way outside
`[0, 8192)`, IndexKernel asserts.

`Alibaba-NLP/new-impl` has not been touched since August 2024 (commit
`40ced75c3017eb27626c9d4ea981bde21a2662f4` is the tip of `main`), so no fix
is coming from the model side.

### Why `transformers==4.57.6` is unaffected

The 4.x loader allocates buffers directly on the target device with
`torch.arange(...)`, so `persistent=False` just means "don't save in the
state dict" — the values survive load correctly. v5 is the first line that
treats `persistent=False` as "skip re-init."

### Evidence

- Model discussion (names the symptom, gives a runtime patch):
  https://huggingface.co/Alibaba-NLP/gte-multilingual-base/discussions/30
- Upstream tracker (reproduces on this exact model):
  https://github.com/huggingface/transformers/issues/43950 — confirms
  `transformers==4.57.6` unaffected
- Companion tracker:
  https://github.com/huggingface/transformers/issues/44534 — reproduces on
  `transformers==5.3.0`
- Release blog describing the loading refactor:
  https://huggingface.co/blog/transformers-v5

## Mitigation: pin in two places

We pin in both `pyproject.toml` and `Dockerfile`. Two pins sounds redundant,
but they protect against different failure modes:

- **`pyproject.toml`** — defines the constraint for library users and for
  `uv lock`. This is where the contract lives.
- **`Dockerfile` explicit `pip install "transformers>=4.46,<5"
  sentence-transformers>=5.0,<5.2` after `pip install .`** — NGC 25.03
  ships no `transformers` and no `sentence-transformers`, so pip resolves
  them from scratch. The explicit install layer ensures the image is
  reproducible even if someone loosens `pyproject.toml` later without
  understanding why the cap exists.

A third guard is a build-time `RUN python -c "import transformers; assert
transformers.__version__.startswith('4.')"` — matches the existing numpy
guardrail pattern. Cheap; fails the build loudly instead of shipping a
broken image.

### Why these specific numbers

- `transformers>=4.46` (was `>=4.41`): no correctness reason for 4.46
  specifically; picks up the preprocessor/video_preprocessor_config probe
  path our startup log shows sentence-transformers issuing. Stays on a
  supported branch.
- `transformers<5`: every v5 release so far (5.0, 5.1, 5.2, 5.3) has the
  regression per the upstream trackers.
- `sentence-transformers>=3.0,<5.2` (pyproject) / `>=5.0,<5.2` (Dockerfile):
  v5.2 introduced "joint transformers v4/v5 compatibility" and starts
  pulling transformers v5 by default on fresh resolves. Staying at ≤5.1.x
  keeps the transitive resolution on transformers 4.x even if someone runs
  `pip install -U sentence-transformers`. pyproject accepts older ST for
  library consumers; Dockerfile tightens to `>=5.0` because that's what the
  container was built against.

## Rejected alternatives

1. **Runtime `register_buffer(..., persistent=True)` patch at load time.**
   Published in discussions/30 and reproduced here:
   ```python
   inner = st_model._first_module().auto_model
   emb = inner.embeddings
   max_pos = emb.position_ids.size(0)
   emb.register_buffer(
       "position_ids",
       torch.arange(max_pos, device=emb.position_ids.device),
       persistent=True,
   )
   ```
   Rejected because it depends on internal attribute paths
   (`_first_module().auto_model.embeddings.position_ids`) that have
   already moved between sentence-transformers 3.x and 5.x. Adds a
   load-time dependency on transformers internals just to sidestep a pin.
   Keep it documented as a break-glass option for the day we *must* run on
   transformers v5.

2. **Switch base image (NGC 25.01, pytorch/pytorch:2.6, …).** Doesn't fix
   anything — the regression is in `transformers`, not the image.

3. **`code_revision` pin on `Alibaba-NLP/new-impl`.** `40ced75` *is* the
   tip; there is no older-known-good that also works with current
   sentence-transformers. Nothing to pin to.

4. **Default to chunk-level embedding.** Was considered when we still
   thought this was a sequence-length bug. Chunk level happens to bound
   inputs to ≤1024 tokens, which would mask the 8192-overrun hypothesis.
   It does not fix the v5 regression — chunk level also hits modeling.py
   and also fails.

## Follow-ups

- Watch https://github.com/huggingface/transformers/issues/43950 — if a v5
  fix ships, relax the pin to `transformers!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*`
  or similar and re-test.
- If we ever replace `Alibaba-NLP/gte-multilingual-base` with a maintained
  embedder (Impresso team may pick a different model in a future revision),
  drop the cap entirely. The pin's only purpose is this model's `new-impl`
  modeling code.
- If we're ever forced onto transformers v5 (transitive pull from another
  dep), apply the runtime `register_buffer(..., persistent=True)` patch
  above, scoped to `load_model` in `src/impresso_text_embedder/model.py`
  with a version check so it no-ops on 4.x.

## Reproduction commands

On the debug pod before the pin landed, the following all reproduced the
crash:

```bash
# Our CLI on BL / BNL:
CUDA_LAUNCH_BLOCKING=1 impresso-embed-create --provider BL  --input-bucket 122-rebuilt-final --output-bucket 141-processed-data-staging --limit 1 --batch-size 4 --force
CUDA_LAUNCH_BLOCKING=1 impresso-embed-create --provider BNL --input-bucket 122-rebuilt-final --output-bucket embedding-test          --limit 2 --batch-size 2

# Plain SentenceTransformer on 6 tokens:
make setup-hf-model
```

After `pip install 'transformers==4.57.6'` (smoke-test only, before pin was
committed):

```bash
# OK: DOWNLOADING THE HUGGINGFACE MODEL DONE
```

Which is what established the cap as correct.
