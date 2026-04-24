"""Unit tests for `impresso_text_embedder.accel`."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from impresso_text_embedder import accel


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear the module's lru_cache between tests so each re-detects."""
    accel.detect_profile.cache_clear()
    accel.has_xformers.cache_clear()
    yield
    accel.detect_profile.cache_clear()
    accel.has_xformers.cache_clear()


@patch("torch.cuda.is_available", return_value=False)
def test_detect_profile_no_cuda(_):
    p = accel.detect_profile()
    assert p.name == "CPU"
    assert p.default_batch_size == 8


@patch("torch.cuda.get_device_capability", return_value=(8, 0))
@patch("torch.cuda.is_available", return_value=True)
def test_detect_profile_a100(_a, _b):
    p = accel.detect_profile()
    assert p.name == "A100"
    assert p.default_batch_size == 64


@patch("torch.cuda.get_device_capability", return_value=(9, 0))
@patch("torch.cuda.is_available", return_value=True)
def test_detect_profile_hopper(_a, _b):
    p = accel.detect_profile()
    assert p.name == "H100/H200"
    assert p.default_batch_size == 128


@patch("torch.cuda.get_device_capability", return_value=(7, 5))
@patch("torch.cuda.is_available", return_value=True)
def test_detect_profile_unknown_cuda(_a, _b):
    p = accel.detect_profile()
    assert p.name == "Unknown-CUDA"
    assert p.default_batch_size == 32


def test_has_xformers_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "xformers", None)
    assert accel.has_xformers() is False


def test_has_xformers_present(monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, "xformers", types.ModuleType("xformers"))
    assert accel.has_xformers() is True
