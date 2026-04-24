"""Per-file timing and GPU-utilization helpers.

Usage from :mod:`pipeline`::

    stats = StageTimer()
    with stats.stage("download"):
        ...
    with stats.stage("encode"), GpuSampler() as gpu:
        ...
    log.info(format_stats_line(stats, gpu_samples={"encode": gpu.summary()}))

Both ``StageTimer`` and ``GpuSampler`` no-op cleanly when their dependencies
are missing (``pynvml`` absent, no CUDA device). See
``.progress/io-throughput/notes.md`` for the rationale.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

log = logging.getLogger(__name__)


class StageTimer:
    """Accumulate wall-clock time per named stage."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self._totals[name] = self._totals.get(name, 0.0) + (time.monotonic() - start)

    @property
    def totals(self) -> dict[str, float]:
        return dict(self._totals)


@dataclass
class GpuSummary:
    """Summary of GPU utilization over a sampling window."""

    samples: int = 0
    mean: float = 0.0
    p10: float = 0.0
    p50: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "mean": round(self.mean, 1),
            "p10": round(self.p10, 1),
            "p50": round(self.p50, 1),
        }


class GpuSampler:
    """Background thread that polls GPU SM utilization at ``hz`` Hz.

    Used as a context manager around the encode window::

        with GpuSampler() as gpu:
            model.encode(...)
        summary = gpu.summary()

    No-ops silently (``samples == 0``) if ``pynvml`` is not importable, if
    CUDA isn't present, or if device initialization fails.
    """

    def __init__(self, device_index: int = 0, hz: float = 2.0) -> None:
        self._device_index = device_index
        self._interval = 1.0 / hz if hz > 0 else 0.5
        self._samples: list[int] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle = None
        self._pynvml = None

    def __enter__(self) -> GpuSampler:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            self._pynvml = pynvml
        except Exception as exc:  # pragma: no cover - depends on deploy env
            log.debug("GpuSampler disabled (pynvml unavailable: %s)", exc)
            return self

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="gpu-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 4)
            self._thread = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass
            self._pynvml = None
            self._handle = None

    def _run(self) -> None:  # pragma: no cover - thread + pynvml side-effects
        assert self._pynvml is not None
        while not self._stop.is_set():
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(int(util.gpu))
            except Exception as exc:
                log.debug("GpuSampler poll failed: %s", exc)
                break
            self._stop.wait(self._interval)

    def summary(self) -> GpuSummary:
        if not self._samples:
            return GpuSummary()
        xs = sorted(self._samples)
        return GpuSummary(
            samples=len(xs),
            mean=statistics.fmean(xs),
            p10=xs[max(0, len(xs) // 10 - 1)] if len(xs) >= 10 else xs[0],
            p50=statistics.median(xs),
        )


def format_stats_line(
    prefix: str,
    timer: StageTimer,
    *,
    records: int,
    gpu: GpuSummary | None = None,
    filter_counter: Mapping[str, int] | None = None,
) -> str:
    """Compact one-line log summary for a completed file.

    When ``filter_counter`` is non-empty, a ``skipped=N (reason=N ...)`` clause
    is inserted between ``records=…`` and the stage timings. Reasons are sorted
    alphabetically for stable log diffing. When the counter is missing, empty,
    or sums to zero, nothing is inserted.
    """
    parts = [prefix, f"records={records}"]
    if filter_counter:
        total = sum(filter_counter.values())
        if total > 0:
            breakdown = " ".join(
                f"{reason}={n}" for reason, n in sorted(filter_counter.items()) if n > 0
            )
            parts.append(f"skipped={total} ({breakdown})")
    for name, seconds in timer.totals.items():
        parts.append(f"{name}_s={seconds:.2f}")
    if gpu is not None and gpu.samples > 0:
        parts.append(
            f"gpu_util_mean={gpu.mean:.1f}% p10={gpu.p10:.1f}% n={gpu.samples}"
        )
    return " ".join(parts)


__all__ = ["StageTimer", "GpuSampler", "GpuSummary", "format_stats_line"]
