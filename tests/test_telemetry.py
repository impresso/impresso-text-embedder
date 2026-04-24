from __future__ import annotations

import sys
import time
from collections import Counter

import pytest

from impresso_text_embedder import telemetry as t


class TestStageTimer:
    def test_accumulates_time_per_stage(self):
        timer = t.StageTimer()
        with timer.stage("a"):
            time.sleep(0.01)
        with timer.stage("b"):
            time.sleep(0.005)
        totals = timer.totals
        assert set(totals) == {"a", "b"}
        assert totals["a"] >= 0.009
        assert totals["b"] >= 0.004
        assert totals["a"] > totals["b"]

    def test_repeated_stage_accumulates(self):
        timer = t.StageTimer()
        for _ in range(3):
            with timer.stage("encode"):
                time.sleep(0.002)
        assert timer.totals["encode"] >= 0.005

    def test_totals_returns_copy(self):
        timer = t.StageTimer()
        with timer.stage("x"):
            pass
        snapshot = timer.totals
        snapshot["x"] = 999.0
        assert timer.totals["x"] != 999.0

    def test_exception_still_records_stage(self):
        timer = t.StageTimer()
        with pytest.raises(RuntimeError):
            with timer.stage("boom"):
                raise RuntimeError("no")
        assert "boom" in timer.totals


class TestGpuSamplerNoop:
    def test_no_pynvml_returns_empty_summary(self, monkeypatch):
        # Pretend pynvml is missing.
        monkeypatch.setitem(sys.modules, "pynvml", None)
        with t.GpuSampler() as gpu:
            time.sleep(0.01)
        summary = gpu.summary()
        assert summary.samples == 0
        assert summary.mean == 0.0

    def test_summary_with_injected_samples(self):
        gpu = t.GpuSampler()
        gpu._samples = [10, 90, 50, 30, 70]
        s = gpu.summary()
        assert s.samples == 5
        assert 10 <= s.mean <= 90
        assert s.p50 == 50


class TestFormatStatsLine:
    def test_includes_all_stages_and_gpu(self):
        timer = t.StageTimer()
        with timer.stage("download"):
            pass
        with timer.stage("encode"):
            pass
        summary = t.GpuSummary(samples=20, mean=87.5, p10=60.0, p50=90.0)
        line = t.format_stats_line("file=SNL/EXP/EXP-1912", timer, records=42, gpu=summary)
        assert line.startswith("file=SNL/EXP/EXP-1912 records=42")
        assert "download_s=" in line
        assert "encode_s=" in line
        assert "gpu_util_mean=87.5%" in line
        assert "p10=60.0%" in line
        assert "n=20" in line

    def test_omits_gpu_block_when_no_samples(self):
        timer = t.StageTimer()
        with timer.stage("encode"):
            pass
        line = t.format_stats_line("x", timer, records=1, gpu=t.GpuSummary())
        assert "gpu_util" not in line

    def test_omits_gpu_block_when_gpu_none(self):
        timer = t.StageTimer()
        line = t.format_stats_line("x", timer, records=0, gpu=None)
        assert "gpu_util" not in line

    def test_includes_filter_counter_sorted(self):
        timer = t.StageTimer()
        with timer.stage("encode"):
            pass
        counter = Counter({"too_short": 6, "content_type": 2})
        line = t.format_stats_line(
            "x", timer, records=100, gpu=None, filter_counter=counter
        )
        # Total + reasons sorted alphabetically.
        assert "skipped=8 (content_type=2 too_short=6)" in line
        # Appears between records=… and the stage timings.
        skipped_idx = line.index("skipped=")
        encode_idx = line.index("encode_s=")
        records_idx = line.index("records=")
        assert records_idx < skipped_idx < encode_idx

    def test_omits_filter_counter_when_empty(self):
        timer = t.StageTimer()
        line = t.format_stats_line(
            "x", timer, records=0, gpu=None, filter_counter=Counter()
        )
        assert "skipped=" not in line

    def test_omits_filter_counter_when_none(self):
        timer = t.StageTimer()
        line = t.format_stats_line("x", timer, records=0, gpu=None)
        assert "skipped=" not in line

    def test_omits_filter_counter_when_all_zero(self):
        timer = t.StageTimer()
        line = t.format_stats_line(
            "x", timer, records=0, gpu=None, filter_counter=Counter({"a": 0, "b": 0})
        )
        assert "skipped=" not in line
