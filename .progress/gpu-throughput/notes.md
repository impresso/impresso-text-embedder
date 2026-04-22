# Model & encoder — A100 optimizations

## Scope

Step 4 lands a working model wrapper and encoder that runs correctly on A100 and degrades cleanly on CPU for CI. Measurement on a real A100 happens at deploy time; the wrapper's defaults are chosen so that measurement is unobstructed.

## Decisions

### bfloat16 via autocast
On CUDA we wrap `model.encode(...)` with `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Reasons:
- A100 has native bf16 tensor cores; dense matmuls get ~2× vs fp32.
- Autocast is model-agnostic and works with `trust_remote_code=True` custom modeling without monkey-patching.
- No loss-scaling headaches (unlike fp16); stable for encoder-only inference.
- We do not force `.to(torch.bfloat16)` on the module itself — autocast is enough and leaves weights in fp32 for norms/layernorms that prefer it.

Fallback on CPU / non-CUDA: no autocast. `gte-multilingual-base` runs in fp32 on CPU (slow, but correct; fine for tests).

### xformers + unpadding — deferred, opt-in via extras
The model card for `Alibaba-NLP/gte-multilingual-base` recommends xformers + unpadding through `Alibaba-NLP/new-impl`. That path requires specific `model_kwargs` at load time and modeling-code cooperation. Wiring it is worth a follow-up step once we can measure delta on a real A100; not worth guessing at now.

- `pyproject.toml` gets an `accel` extra that installs `xformers` so users on A100 can `uv sync --extra accel` without us forcing it into the base install (where it may fail to build).
- A future step (see `plan.md`) will decide whether `attn_implementation` / custom `model_kwargs` are actually forwarded through the `SentenceTransformer` wrapper.

### flash-attn — out of scope for step 4
`Alibaba-NLP/gte-multilingual-base` does not document `attn_implementation="flash_attention_2"` support. Its recommended accelerator is xformers. Revisit only if xformers is shown insufficient.

### `model.eval()` + `torch.inference_mode()`
Always. No gradient tracking during inference.

### Batch size
CLI exposes `--batch-size`; the encoder function takes it as an argument. **Starting guesses** (to tune on real A100, then fix in CLAUDE.md):
- text-level: 64
- sentence-level: 128
- chunk-level: 128

These are placeholders; an 80 GB A100 with bf16 will support more. Measurement step fills in real numbers.

### Trust remote code
The model requires `trust_remote_code=True`. Documented in the loader. Not a tunable.

## Non-decisions deliberately left open

- Exact xformers flags and model_kwargs passthrough.
- Whether to `torch.compile(model)` — depends on whether ST's encode path is traceable. Likely fragile; benchmark before enabling.
- DDP / multi-GPU — explicit non-goal per CLAUDE.md.

## API shape for step 4

```python
from impresso_text_embedder.model import load_model, encode_texts

model = load_model(name="Alibaba-NLP/gte-multilingual-base", revision=None)
vecs = encode_texts(model, texts, batch_size=64, normalize=True)  # np.ndarray [N, D]
```

No singleton, no global state. Callers own the instance.

## What's tested now

- `load_model` is called with the correct ST constructor args (mocked).
- Device selection logic picks CUDA if `torch.cuda.is_available()`, CPU otherwise.
- `encode_texts` forwards expected kwargs to `model.encode` and respects `batch_size`.
- Autocast context is entered iff device is CUDA.

A heavyweight integration test that actually downloads the model is **not** in this step's test gate; it would slow CI and add flakiness for no unique assertion.
