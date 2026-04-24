"""Shared pytest fixtures for impresso-text-embedder tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_rcp_scratch(tmp_path, monkeypatch):
    """Make ``logging_setup``'s default log base resolve to a tmp dir.

    Production defaults to ``/rcp-scratch`` and fails hard when the PVC is
    not mounted. Tests that invoke ``cli.create.main`` would otherwise need
    to pass ``--log-dir`` everywhere; this fixture swaps the base per test
    so existing tests keep working unchanged. Tests that specifically
    exercise the "missing path" branch patch ``DEFAULT_RCP_SCRATCH``
    themselves inside a ``with`` block, which takes precedence.
    """
    from impresso_text_embedder import logging_setup

    fake_root = tmp_path / "rcp-scratch"
    fake_root.mkdir()
    monkeypatch.setattr(logging_setup, "DEFAULT_RCP_SCRATCH", fake_root)
