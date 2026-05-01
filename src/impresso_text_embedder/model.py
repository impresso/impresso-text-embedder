"""Model loading and encoding wrapper.

A thin layer over ``sentence_transformers.SentenceTransformer`` that:
  * loads with ``trust_remote_code=True`` (required by ``gte-multilingual-base``);
  * picks CUDA when available;
  * wraps ``encode(...)`` in a bf16 autocast on CUDA (Ampere/Hopper tensor cores);
  * leaves batch size to the caller.

See ``.history/gpu-throughput/notes.md`` for the rationale.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .accel import has_xformers, log_profile

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_MODEL_REVISION = "f7d567e"


def select_device() -> str:
    """Return ``"cuda"`` if a CUDA device is visible, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    name: str = DEFAULT_MODEL_NAME,
    revision: str | None = DEFAULT_MODEL_REVISION,
    device: str | None = None,
    *,
    use_xformers: bool | None = None,
    unpad_inputs: bool | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer model pinned to ``revision`` (if given).

    On CUDA, the model is left in fp32 and bf16 is applied at encode time via
    ``torch.autocast`` (see :func:`encode_texts`). That keeps LayerNorm in fp32,
    which is safer than casting the whole module.

    ``use_xformers`` and ``unpad_inputs`` independently gate the two
    ``config_kwargs`` Alibaba's new-impl modeling file reads from
    ``self.config``. ``None`` (default) auto-detects — both enabled when the
    device is CUDA and ``xformers`` is importable, both disabled otherwise.
    Explicit ``True`` requires CUDA + ``xformers`` and raises
    :class:`RuntimeError` if either is missing (no silent fallback);
    explicit ``False`` disables the flag unconditionally. Callers that want
    to bisect numerical drift (e.g. the CLI's ``--attention`` /
    ``--unpad-inputs`` flags) pass explicit booleans.
    """
    from sentence_transformers import SentenceTransformer

    resolved_device = device or select_device()
    log_profile()
    log.info("Loading SentenceTransformer %s@%s on %s", name, revision or "default", resolved_device)

    auto = resolved_device.startswith("cuda") and has_xformers()
    if use_xformers is None:
        use_xformers = auto
    elif use_xformers:
        if not resolved_device.startswith("cuda"):
            raise RuntimeError(
                f"use_xformers=True requires a CUDA device, got device={resolved_device!r}"
            )
        if not has_xformers():
            raise RuntimeError(
                "use_xformers=True but the xformers package is not importable"
            )
    if unpad_inputs is None:
        unpad_inputs = auto

    st_kwargs: dict[str, Any] = {
        "model_name_or_path": name,
        "trust_remote_code": True,
        "revision": revision,
        "device": resolved_device,
    }
    # Two independent config flags. Alibaba's new-impl modeling file reads
    # them from self.config, not __init__ kwargs — and ST v5 pre-loads the
    # config and passes it explicitly to from_pretrained, which skips HF's
    # kwarg-to-config routing. So they must go via config_kwargs, not
    # model_kwargs. Combinations are intentionally permitted (e.g. unpad
    # without xformers) so an operator can bisect numerical drift; the
    # modeling file decides whether the combination is meaningful.
    config_kwargs: dict[str, Any] = {}
    if unpad_inputs:
        config_kwargs["unpad_inputs"] = True
    if use_xformers:
        config_kwargs["use_memory_efficient_attention"] = True
    if config_kwargs:
        st_kwargs["config_kwargs"] = config_kwargs

    model = SentenceTransformer(**st_kwargs)
    model.eval()
    _assert_built_in_normalize(model, name, revision)
    log.info(
        "Model loaded (device=%s use_xformers=%s unpad_inputs=%s)",
        resolved_device,
        use_xformers,
        unpad_inputs,
    )
    return model


def _assert_built_in_normalize(
    model: SentenceTransformer, name: str, revision: str | None
) -> None:
    """Refuse to run on a model that doesn't end with a Normalize module.

    The pipeline assumes encoder outputs are unit-norm — the cosine validation
    contract (``--tol 1e-4``) and ``MeanPoolStrategy``'s direction-only
    averaging both depend on it. Failing at load time turns that assumption
    into a hard invariant. See ``.history/normalize-flag-removal/notes.md``.
    """
    from sentence_transformers.models import Normalize

    last = model[-1]
    if not isinstance(last, Normalize):
        raise RuntimeError(
            f"Model {name}@{revision or 'default'} does not end with a "
            f"sentence_transformers.models.Normalize module "
            f"(last module is {type(last).__name__}). The pipeline requires "
            f"unit-norm encoder outputs; see "
            f".history/normalize-flag-removal/notes.md."
        )


@contextlib.contextmanager
def _cuda_bf16_autocast(device: str, enabled: bool) -> Iterator[None]:
    """Enable bf16 autocast on CUDA when ``enabled`` is True; no-op otherwise."""
    if enabled and device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    show_progress_bar: bool = False,
    *,
    precision: str = "bf16",
) -> np.ndarray:
    """Encode ``texts`` into a numpy array of shape ``[len(texts), D]``.

    With ``precision="bf16"`` (default) and a CUDA device, the encode runs
    under bf16 autocast + ``torch.inference_mode()``. With ``precision="fp32"``
    the autocast scope is skipped (model weights are already fp32). On CPU
    autocast is always skipped. Output vectors are unit-norm because the
    model is required to ship a final ``Normalize`` module (asserted at load
    time); we do not request a redundant L2 here.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    device = _current_device(model)
    autocast_enabled = precision == "bf16"
    with torch.inference_mode(), _cuda_bf16_autocast(device, autocast_enabled):
        out = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    return _as_float32(out)


def _current_device(model: SentenceTransformer) -> str:
    try:
        first_param_device = next(model.parameters()).device
    except StopIteration:
        return "cpu"
    return "cuda" if first_param_device.type == "cuda" else first_param_device.type


def _as_float32(out: Any) -> np.ndarray:
    arr = np.asarray(out)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr
