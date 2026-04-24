"""GPU profile detection for A100 / H100 (incl. H200) / CPU.

Profile carries only the default batch size and a display name. Attention
kernel selection is transparent: xformers' ``memory_efficient_attention``
dispatches to FA3 on Hopper and FA2 on Ampere on its own, based on the
device + dtype of Q/K/V. See ``.progress/gpu-profiles/notes.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    name: str
    default_batch_size: int
    notes: str


_A100 = Profile(
    name="A100",
    default_batch_size=64,
    notes="Ampere cc 8.0; xformers → FA2",
)
_HOPPER = Profile(
    name="H100/H200",
    default_batch_size=128,
    notes="Hopper cc 9.0; xformers → FA3",
)
_UNKNOWN_CUDA = Profile(
    name="Unknown-CUDA",
    default_batch_size=32,
    notes="Unrecognised compute capability; using conservative defaults",
)
_CPU = Profile(
    name="CPU",
    default_batch_size=8,
    notes="No CUDA device visible",
)


@lru_cache(maxsize=1)
def detect_profile() -> Profile:
    """Return the profile that matches the first visible CUDA device.

    Cached so repeated calls don't re-probe the device or re-log.
    """
    if not torch.cuda.is_available():
        return _CPU
    cc = torch.cuda.get_device_capability(0)
    if cc == (8, 0):
        return _A100
    if cc == (9, 0):
        return _HOPPER
    log.warning(
        "Unrecognised CUDA compute capability %s — falling back to conservative profile",
        cc,
    )
    return _UNKNOWN_CUDA


@lru_cache(maxsize=1)
def has_xformers() -> bool:
    """Return True if ``xformers`` is importable.

    The Alibaba-NLP custom modeling file only enables its fast attention
    path when the two ``config_kwargs`` ``unpad_inputs`` and
    ``use_memory_efficient_attention`` are set on the model config; those
    in turn require xformers to be installed at runtime.
    """
    try:
        import xformers  # noqa: F401
    except ImportError:
        return False
    return True


def log_profile(profile: Profile | None = None) -> None:
    """Emit a single INFO line describing the active profile + xformers state."""
    p = profile or detect_profile()
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    log.info(
        "Detected %s (profile=%s, cc=%s), default batch=%d, xformers=%s",
        device_name,
        p.name,
        f"{cc[0]}.{cc[1]}" if cc else "n/a",
        p.default_batch_size,
        "yes" if has_xformers() else "no",
    )
