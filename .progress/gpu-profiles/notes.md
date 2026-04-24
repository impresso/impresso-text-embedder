# GPU profiles (A100 + H100) — design and open questions

Status: **todo**. This doc captures the design before implementation so the choices are visible and revisable. Decisions that are still open are marked **OPEN**; fill them in as they land.

## Problem

Today the code is tuned for A100 only: bf16 autocast, no explicit attention backend, `--batch-size` default 64, and `CLAUDE.md` says "A100 only". We want to run on H100 too (Run:AI on EPFL RCP exposes V100/A100/H100/H200) without forking the codebase. H100 offers 2–3× bf16 throughput over A100 for transformer encoders at 8192 seq len, driven mainly by 4th-gen tensor cores, 2.15× HBM bandwidth, and Flash-Attention 2/3. The gain is worth the small amount of plumbing required.

## Scope

- **In**: A100 and H100, single-GPU, forward-only inference for `gte-multilingual-base`.
- **Out (deferred levers)**:
  - **FP8 via TransformerEngine** — another potential ~2× on H100 but requires TE integration or TensorRT-LLM. Non-trivial. Revisit when BF16 throughput on H100 stops being the bottleneck.
  - **V100** — no bf16 tensor cores. Different precision strategy (fp16 with loss-scaling or fp32). Not planned.
  - **H200** — same compute as H100, more memory bandwidth and capacity. Our model is ~600 MB in bf16; capacity is not binding. If H100 is unavailable we can run on H200 transparently (profile picks H100 settings from cc `(9,0)`), but it's not a dedicated profile.
  - **Multi-GPU / DDP** — already out of scope per `CLAUDE.md`.

## Decision: runtime detection, not build-time selection

Two ways to route to the right profile:

1. **Runtime detection** — one image, one binary. At startup, read `torch.cuda.get_device_capability()` and pick the profile. Log the detected arch on the first line.
2. **Build-time selection** — image is labeled `…-h100` or `…-a100`; code trusts the label and fails loudly if the GPU doesn't match.

Picked **(1) runtime detection + explicit startup log line**. Reasons:
- Run:AI pod may land on an unexpected node type; runtime detection degrades gracefully, build-time silently misbehaves.
- One code path is easier to test and reason about.
- The startup log (`Detected NVIDIA H100 (cc 9.0), attn=flash_attention_2, default batch=128`) makes the chosen profile visible in every run's logs, which is enough of a safety rail.

The Docker image *tag* still carries the arch (`:<v>-h100`) because the image's **dep set** differs (flash-attn wheel for H100). So we ship two images, but the code inside both is identical and self-configuring.

## Profiles

Simplified after the research above: attention kernel is xformers everywhere. Profile only carries the batch-size default and a display name.

| Profile | cc | Default batch (text level) | Notes |
|---|---|---|---|
| A100 | `(8, 0)` | 64 | xformers → FA2 |
| H100 | `(9, 0)` | **OPEN** — start 128, measure | xformers → FA3 (transparent) |
| H200 | `(9, 0)` | Same as H100 | Reports same cc as H100; same kernel |
| Fallback | anything else | 32 | xformers → its own MEA kernel, still a win |

Batch-size defaults are per-profile *and* per embedding level. The `create` CLI already supports larger batches for sentence/chunk levels than text; the profile only sets the *text-level* default, others derive from it.

## Resolved: `gte-multilingual-base` does NOT honor `attn_implementation`

Confirmed via the model's own HF discussion and the `new-impl` repo (Alibaba's custom modeling file that `trust_remote_code` loads). Calling

```python
AutoModel.from_pretrained(path, trust_remote_code=True, attn_implementation="flash_attention_2")
```

raises:

```
ValueError: NewModel does not support Flash Attention 2.0 yet. Please request to add support where the model is hosted.
```

The entire HF-standard `attn_implementation` / SDPA-dispatch path is closed for this model. That collapses the "per-arch attn_impl routing" design I sketched earlier.

### The real acceleration path: xformers + unpadding

Alibaba's documented path, from their [new-impl README](https://huggingface.co/Alibaba-NLP/new-impl) and reiterated on the [model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base):

```python
model = AutoModel.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    unpad_inputs=True,
    use_memory_efficient_attention=True,
    torch_dtype=torch.float16,  # or torch.bfloat16
).to(device)
```

`unpad_inputs` and `use_memory_efficient_attention` are **config attributes** read by the custom modeling file off `self.config` — not `__init__` kwargs on `NewModel`. The raw `AutoModel.from_pretrained(..., unpad_inputs=True, ...)` call above works only because HF's `from_pretrained` pops config-shaped kwargs onto the config object **when it loads the config itself**. That kwarg-to-config routing is the load-bearing bit — and sentence-transformers v5 removes it.

**Recorded: route via `config_kwargs`, not `model_kwargs`.** Sentence-transformers v5's `Transformer._load_model` pre-loads the config and then calls `model_cls.from_pretrained(path, config=config, **model_kwargs)`. When `config=` is passed explicitly, HF's `from_pretrained` skips the kwarg-to-config pop, so `unpad_inputs` and `use_memory_efficient_attention` survive into `cls(config, **model_kwargs)` → `NewModel.__init__(unpad_inputs=True, ...)` → `TypeError`. The fix is to pass the two flags via `config_kwargs={...}` on `SentenceTransformer(...)`; sentence-transformers forwards those to `AutoConfig.from_pretrained(path, **config_kwargs)`, which sets them as attributes on the config object via `PretrainedConfig.__init__`'s catch-all `**kwargs`. `NewModel(config)` then reads `self.config.unpad_inputs` and takes the fast path.

Confirmed on a Run:AI A100 pod with sentence-transformers 5.4.1 + transformers shipping in NGC 25.03 on 2026-04-23: the `model_kwargs` path raised `TypeError: NewModel.__init__() got an unexpected keyword argument 'unpad_inputs'`; the `config_kwargs` path loads cleanly. Equivalent to editing `config.json` in the HF cache, without touching on-disk state.

### What xformers gives us across archs — for free

From xformers' dispatch rules (confirmed in xformers 0.0.35 docs and the FA3 paper's xformers integration note):

- xformers' `memory_efficient_attention` inspects device + dtype and picks the best kernel. No per-arch code on our side.
- **H100**: dispatches to FlashAttention-3 (Hopper-specialized, ~1.5–2× over FA2 at fp16, up to ~750 TFLOPs/s). The FA library itself inspects hardware IDs to pick FA2 (Ampere) vs FA3 (Hopper).
- **A100**: dispatches to FlashAttention-2.
- **V100** (and other older archs): dispatches to xformers' own memory-efficient kernel — Alibaba explicitly claims this gives "significant acceleration on old devices like the V100". Doesn't change our scope (still out), but means the fallback profile degrades gracefully instead of erroring.

### Consequences for the plan

Most of the per-arch plumbing I sketched earlier dissolves. Updated design:

1. **One code path, no attn-impl routing.** Both A100 and H100 load the model with `unpad_inputs=True, use_memory_efficient_attention=True`. xformers + FA handle the rest. `accel.py` no longer needs an `attn_impl` field on the profile.
2. **One extras group, not two.** `accel = ["xformers>=…"]`. Possibly `flash-attn` as a sibling dep because xformers' FA3 dispatch on Hopper depends on the `flash-attn` package being installed and recent enough (≥ 2.7 for FA3 support). **OPEN**: confirm whether xformers bundles its own FA3 path or needs `flash-attn` installed alongside. If separate: `accel = ["xformers>=…", "flash-attn>=2.7"]`.
3. **Two Docker images are still useful** but for thinner reasons: pinning `flash-attn` wheels that match CUDA/torch is expensive, and if the wheel we need doesn't exist we want the build to be per-arch rather than the fattest common denominator. Could also collapse to one image if the same wheel works on both — decide once the NGC torch version is pinned.
4. **Profile still exists**, but only carries `name` and `default_batch_size` (plus any diagnostics). No attention routing.
5. **Numerical-drift validation is still required** — the attention kernel xformers dispatches to differs between A100 and H100 (FA2 vs FA3), and FA3 uses asynchronous tensor-core pathways with different rounding. The `--tol 1e-4` cosine check is exactly the right guard. Verdict to be recorded after real-GPU runs.
6. **Precision decision needs re-visiting.** `CLAUDE.md` currently says "weights stay fp32, bf16 via autocast around encode" specifically to keep LayerNorm stable on the HF-generic path. Alibaba's docs load weights in `float16` / `bfloat16` directly. xformers' memory-efficient kernel *requires* half-precision inputs to fire the FA path — autocast-only-with-fp32-weights likely sidesteps the fast path. **OPEN**: test whether passing `torch_dtype=torch.bfloat16` to `from_pretrained` breaks the LayerNorm concern that drove the autocast-only choice. If not, switch to bf16 weights + no autocast, matching Alibaba's recipe.

### Sources

- Model discussion where the error is reported: https://huggingface.co/Alibaba-NLP/gte-multilingual-base/discussions/8
- Alibaba's modeling repo with the official recipe: https://huggingface.co/Alibaba-NLP/new-impl (section "Recommendation: Enable Unpadding and Acceleration with xformers")
- Model card: https://huggingface.co/Alibaba-NLP/gte-multilingual-base
- FlashAttention-3 paper (Hopper kernel, xformers integration): https://arxiv.org/abs/2407.08608
- xformers dispatch docs: https://facebookresearch.github.io/xformers/components/ops.html

## Packaging

One extras group, just xformers:

```toml
[project.optional-dependencies]
accel = ["xformers>=0.0.28"]
```

**Revised (Q1)**: an earlier version of this note claimed xformers bundles its own FA2/FA3 kernels on PyPI. That was wrong. The correct picture, verified against NGC 25.03's official release notes and the 24.02 syft component manifest:

- **xformers wheels on PyPI (and on PyTorch's cu128 index) ship a CUTLASS-based `memory_efficient_attention` kernel only.** They do not bundle flash-attn.
- **xformers dispatches to FA2 / FA3 when the `flash-attn` package is also installed.** Without `flash-attn`, `memory_efficient_attention` falls back to CUTLASS — still a real speedup, still supports unpadding, but not the Hopper-specialised FA3 kernel.
- **flash-attn** is published separately on PyPI (build from source) and as prebuilt wheels on the Dao-AILab releases page. Installing it is where the real H100 win comes from.

The xformers "3rdParty/flash-attention" submodule only exists in the **source** repo; it's used when building xformers from source so the resulting binary includes FA. **Wheels skip that step** (confirmed by inspection: xformers wheels at `pytorch.org/whl/cu128/xformers/` are `cp39-abi3` slim builds, no FA symbols).

Operational consequence: **both packages need to be installed to get FA2/FA3.** Today we install only xformers (phase 1, CUTLASS kernel, still good). flash-attn is deferred to phase 2, gated on live-GPU measurement.

NGC `pytorch:25.03-py3` does **not** bundle either package. The previous "xformers is in NGC" claim in this file and in the Dockerfile comments was wrong — NGC 25.03's explicit contents list enumerates CUDA, cuDNN, NCCL, TE, RAPIDS, DALI, TensorRT, etc., but not xformers or flash-attn. (For reference, NGC 24.02 shipped `flash-attn 2.4.2` per syft but still no xformers; 25.03 release notes don't mention flash-attn at all, so we can't rely on it being present.)

### What the Dockerfile does today

```dockerfile
RUN pip install --no-cache-dir --no-deps \
      "xformers==0.0.30" \
      --index-url https://download.pytorch.org/whl/cu128
RUN python -c "import xformers; print('xformers', xformers.__version__)"
```

Pinned to 0.0.30 because that's the published match for torch 2.7.0 + CUDA 12.8. `--no-deps` keeps pip from touching NGC's torch (which would break the apex/NCCL ABI stack). The assertion catches a silent regression.

### Deferred lever: add flash-attn alongside

If live-GPU measurement on H100 shows the CUTLASS kernel leaves throughput on the table, add:

```dockerfile
# Prebuilt wheel from Dao-AILab; use the asset URL that matches
# torch 2.7 + CUDA 12.8 + cp312-cp312 + cxx11abiTRUE.
RUN pip install --no-cache-dir --no-deps <flash-attn-wheel-url>
RUN python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
```

This is documented as a deferred lever in `CLAUDE.md` and left out of the first container to keep build time and blast radius small.

### CUDA and torch compatibility

- xformers 0.0.30 → torch 2.7.0 → FA2 2.7.1–2.7.4 → CUDA 12.6.3 or 12.8.0 (from the Medium compatibility guide).
- NGC 25.03 has torch 2.7.0a (alpha) + CUDA 12.8.1. In practice PyTorch maintains ABI compat within a minor, so 0.0.30 should load; if it doesn't, bump to 0.0.31 or 0.0.35.

Base install has no xformers, so a developer running `uv sync` on a laptop (no CUDA) doesn't pull the heavy wheel. Missing xformers degrades gracefully — `accel.has_xformers()` returns False and `model.py` skips the `use_memory_efficient_attention` kwarg — the model's default attention path still runs, just slower.

Sources:
- xformers dispatch table: https://facebookresearch.github.io/xformers/components/ops.html
- xformers wheels at PyTorch cu128 index: https://download.pytorch.org/whl/cu128/xformers/
- NGC PyTorch 25.03 release notes (explicit component list): https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html
- NGC 24.02 syft component manifest (for comparison): https://gist.github.com/jspeed-meyers/5a9d33c7a7fb22c35691763506a299aa
- Compatibility matrix for xformers/torch/CUDA/FA2: https://medium.com/@vici0549/the-definitive-guide-to-pytorch-cuda-and-flash-attention-compatibility-ebec1161ec10
- FA3 CUDA requirement: https://github.com/Dao-AILab/flash-attention

## Dockerfile & Makefile plumbing

Single `Dockerfile`:
```dockerfile
ARG GPU_ARCH=h100
RUN pip install -e .[accel-${GPU_ARCH}]
```

Makefile:
- `GPU_ARCH ?= h100` (flip default from a100 once H100 is validated as the primary target).
- `docker-build-push` tags `$(IMAGE):$(VERSION)-$(GPU_ARCH)`, passes `--build-arg GPU_ARCH`.
- `runai-submit` needs a `GPU_ARCH → nodeSelector` mapping. Run:AI exposes the GPU product via a node label, likely `nvidia.com/gpu.product=NVIDIA-H100-…`. **OPEN**: confirm the exact label values on EPFL RCP.

## Validation — the actual risk

Different attention kernels produce numerically different outputs. The `impresso-embed-validate` cosine-distance-≤-`1e-4` check is the right tool, but it hasn't been exercised across kernels.

Plan:
1. On A100, generate a golden output for a small input with the new profile code (`attn=sdpa`).
2. On H100, run the same input through `attn=flash_attention_2`, validate against the golden.
3. Expected outcome: passes at `1e-4`, not bit-identical. If it passes, record the max observed cosine distance here.
4. If it fails: **OPEN** — options are (a) loosen tol with rationale, (b) force SDPA on both and lose some H100 headroom, (c) store separate goldens per arch.

Don't ship the profile code without running this comparison. Record the outcome here.

## Measurement to record once real-GPU time is available

- **Throughput** (items/sec at each embedding level) on A100 and H100, at multiple batch sizes. The sweet spot on H100 is probably 128–256 for text; measure.
- **Max observed cosine distance** between A100 (xformers→FA2) and H100 (xformers→FA3) outputs on the same input.
- **Startup cost** — model load + first encode — per profile. Expect higher on H100 if flash-attn kernels JIT-compile on first use.
- **bf16-weights vs autocast** (see resolution below): measurement is now confirmation-only, not a precondition for shipping.

## Resolved (Q2): keep `fp32 weights + bf16 autocast`

Researched across PyTorch, Lightning, and HF docs:

- **PyTorch ≥ 1.10** correctly accumulates LayerNorm in fp32 regardless of input dtype. The historical LayerNorm instability that motivated "keep weights in fp32" is already fixed at the framework level.
- **Autocast** keeps weights in fp32 and casts supported ops (matmul, attention) to bf16. Ops without bf16 support silently promote back to fp32. Safer default, smaller blast radius on unknown operators.
- **Explicit `torch_dtype=bfloat16`** casts every parameter. Uniform, faster in theory, but any op without bf16 support breaks or runs slower.
- **xformers' memory_efficient_attention dispatches on Q/K/V tensor dtype**, not on weight dtype. Inside an autocast region, Q/K/V are already bf16 by the time they hit the attention op — so xformers fires the FA2/FA3 kernel either way. Load-time bf16 is not a precondition for the fast path.
- **Memory savings of bf16 weights are ~300 MB** for `gte-multilingual-base`. Irrelevant on A100/H100 80GB.

**Verdict**: our current recipe (fp32 weights + bf16 autocast around `encode`) stays. Alibaba's recommended `torch_dtype=bfloat16` load is one valid path, not a requirement. No change to `embed.py`.

Measurement on real hardware should still verify that the xformers FA kernel actually fires — detectable by checking `torch.cuda.profiler` traces or adding a one-time `xformers.ops.memory_efficient_attention` probe at model-load time. If it doesn't fire in autocast mode for some reason, **then** revisit with bf16-load.

Sources:
- PyTorch autocast docs: https://docs.pytorch.org/docs/stable/amp.html
- Explicit cast vs autocast discussion: https://discuss.pytorch.org/t/bfloat16-training-explicit-cast-vs-autocast/202618
- HF perf docs on bf16 inference: https://huggingface.co/docs/transformers/performance

## A100 → H100 feature-by-feature, relevance to this workload

Workload: forward-only BF16 transformer encoder, sequences up to 8192 tokens, batch 64+. SM matmul *and* HBM bandwidth both matter; FA3 attention kernel matters at long sequences.

| Feature | A100 SXM 80GB | H100 SXM 80GB | H100 gain | Our use | Net relevance |
|---|---|---|---|---|---|
| Architecture (CC) | Ampere, SM 8.0 | Hopper, SM 9.0 | — | Detected at runtime | — |
| BF16 / FP16 Tensor Core (dense) | 312 TFLOPS | 989 TFLOPS | **3.17×** | Main matmul path | **Primary speedup driver** |
| FP8 Tensor Core | — | 1979 TFLOPS (dense) | new | Needs TransformerEngine | Deferred lever |
| FP32 (non-Tensor) | 19.5 TFLOPS | 67 TFLOPS | 3.43× | LayerNorm, softmax residuals (when not fused) | Minor — most fused paths go through TC |
| HBM capacity | 80 GB HBM2e | 80 GB HBM3 | 1× | Model is ~600 MB; 99% headroom on both | None for our model size |
| Memory bandwidth | ~2.0 TB/s | ~3.35 TB/s | **1.67×** | LayerNorm, activations, gather ops | **Real-world throughput floor** |
| L2 cache | 40 MB | 50 MB | 1.25× | Batch residency between layers | Modest |
| Flash-Attention kernel | FA2 | **FA3** (Hopper-specialized) | ~1.5–2× attention | Attention at 8192 seq len | **Large at long seq** |
| TMA (Tensor Memory Accelerator) | — | yes | new | Used by FA3 internally | Transparent via xformers |
| Transformer Engine (FP8 autocast) | — | yes | new | Not wired | Deferred lever |
| NVLink gen / bandwidth | 3rd gen, 600 GB/s | 4th gen, 900 GB/s | 1.5× | Single-GPU; irrelevant | None |
| Thread Block Clusters / DSMEM | — | yes | new | Kernel-level, opaque to PyTorch | None (transparent) |

### Expected end-to-end speedup on our workload: **~2.0–2.8× A100 → H100**

Reasoning:
- Pure-matmul ceiling is 3.17×. We won't hit it because memory bandwidth caps real-world transformer inference before the tensor cores saturate.
- Bandwidth floor is 1.67×. The realistic encode throughput lands between the two, weighted by how matmul-heavy the forward pass is (attention at 8192 seq len is very matmul-heavy, which pulls us toward the ceiling).
- FA3 gives another ~1.5–2× *on the attention sub-op*, which is a large fraction of the 8192-seq-len forward pass. This lifts the effective speedup above a simple memory-bandwidth floor.
- FP8 via TransformerEngine could push to ~3.5–4× on top of bf16 — but that's a real engineering project, not a flag flip. Listed as a deferred lever.

### H200 vs H100 on our workload

Same Hopper compute (989 BF16 TFLOPS). 141 GB HBM3e at 4.8 TB/s. For `gte-multilingual-base`:
- Capacity advantage: **unused**. Our model is ~600 MB; we're not approaching 80 GB.
- Bandwidth advantage: ~1.43×. Translates to maybe 10–20% on bandwidth-bound kernels (LayerNorm, activations). Not worth queuing for unless H100 is unavailable.
- The runtime profile treats H200 as H100 (same CC reports `(9,0)`), which is fine — we get the FA3 kernel automatically.

Sources:
- A100 vs H100 spec comparison: https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100
- NVIDIA Hopper architecture deep-dive: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- H200/H100/A100 comparison: https://modal.com/blog/h200-vs-h100-vs-a100
- FA3 paper with measured TFLOPS: https://arxiv.org/abs/2407.08608

Populate this section, then update `CLAUDE.md`'s "Still open — needs real A100 time" entry to point here instead.

## Links to touch in `CLAUDE.md`

- Remove "A100 only" from **Target hardware**.
- Add to **Decisions recorded**: profile-based backend selection (runtime detection), per-arch batch-size defaults, H100 as the primary target.
- Move FP8 / TransformerEngine from "deferred" language into an explicit deferred lever with a pointer to this note.
- Update the **Commands** example to `GPU_ARCH=h100` defaults.
