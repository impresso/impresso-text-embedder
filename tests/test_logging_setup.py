"""Tests for ``impresso_text_embedder.logging_setup``."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from impresso_text_embedder import logging_setup
from impresso_text_embedder.logging_setup import (
    TqdmLoggingHandler,
    _resolve_log_path,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Each test starts with a clean root logger and restores afterwards."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)


def test_resolve_fails_when_rcp_scratch_missing(tmp_path: Path) -> None:
    # Point the module's default base at a path that does not exist.
    with patch.object(
        logging_setup, "DEFAULT_RCP_SCRATCH", tmp_path / "does-not-exist"
    ):
        with pytest.raises(SystemExit, match="--log-dir"):
            _resolve_log_path(provider="BNL", log_dir=None)


def test_resolve_uses_log_dir_override(tmp_path: Path) -> None:
    now = datetime(2026, 4, 23, tzinfo=timezone.utc)
    path = _resolve_log_path(provider="BNL", log_dir=tmp_path, now=now)
    assert path == tmp_path / "2026-04-23" / "BNL.log"
    assert path.parent.is_dir()


def test_resolve_uses_default_when_rcp_scratch_present(tmp_path: Path) -> None:
    # The autouse conftest fixture already creates tmp_path/rcp-scratch, but we
    # point at a nested subdir to make the assertion independent of it.
    fake_root = tmp_path / "fake-rcp-scratch"
    fake_root.mkdir()
    now = datetime(2026, 4, 23, tzinfo=timezone.utc)
    with (
        patch.object(logging_setup, "DEFAULT_RCP_SCRATCH", fake_root),
        patch("impresso_text_embedder.logging_setup.getpass.getuser", return_value="alice"),
    ):
        path = _resolve_log_path(provider="BNL", log_dir=None, now=now)
    assert path == fake_root / "alice" / "experiments" / "embeddings" / "2026-04-23" / "BNL.log"


def test_configure_logging_writes_info_to_file(tmp_path: Path) -> None:
    path = configure_logging(provider="BNL", log_dir=tmp_path, log_level_file="INFO")
    logging.getLogger("some.module").info("hello file")
    # Force the FileHandler to flush.
    for h in logging.getLogger().handlers:
        h.flush()
    content = path.read_text(encoding="utf-8")
    assert "hello file" in content
    assert "INFO" in content


def test_configure_logging_warning_level_filters_info(tmp_path: Path) -> None:
    path = configure_logging(provider="BNL", log_dir=tmp_path, log_level_file="WARNING")
    log = logging.getLogger("other.module")
    log.info("not written")
    log.warning("written")
    for h in logging.getLogger().handlers:
        h.flush()
    content = path.read_text(encoding="utf-8")
    assert "not written" not in content
    assert "written" in content


def test_tqdm_handler_receives_errors_not_info(tmp_path: Path) -> None:
    configure_logging(provider="BNL", log_dir=tmp_path, log_level_file="INFO")
    with patch("impresso_text_embedder.logging_setup.tqdm.write") as mock_write:
        log = logging.getLogger("yet.another")
        log.info("info message")
        log.warning("warning message")
        log.error("boom")
    # Only ERROR should reach the terminal handler's tqdm.write.
    assert mock_write.call_count == 1
    emitted = mock_write.call_args[0][0]
    assert "boom" in emitted
    assert "ERROR" in emitted


def test_tqdm_handler_default_level_is_error() -> None:
    handler = TqdmLoggingHandler()
    assert handler.level == logging.ERROR


# --- step 18: per-shard log filename ----------------------------------------


def test_resolve_log_path_unchanged_when_no_shard_kwargs(tmp_path: Path) -> None:
    """Existing call sites (no shard kwargs) get `<provider>.log` as before."""
    now = datetime(2026, 4, 23, tzinfo=timezone.utc)
    path = _resolve_log_path(provider="BNL", log_dir=tmp_path, now=now)
    assert path.name == "BNL.log"


def test_resolve_log_path_unchanged_when_num_shards_one(tmp_path: Path) -> None:
    """Default sharding (N=1) keeps the historical filename."""
    now = datetime(2026, 4, 23, tzinfo=timezone.utc)
    path = _resolve_log_path(
        provider="BNL", log_dir=tmp_path, now=now, shard_index=0, num_shards=1
    )
    assert path.name == "BNL.log"


def test_resolve_log_path_includes_shard_suffix_when_n_gt_1(tmp_path: Path) -> None:
    """Multi-shard (N>1) appends `-shard-i-of-N` so concurrent shards don't collide."""
    now = datetime(2026, 4, 23, tzinfo=timezone.utc)
    path = _resolve_log_path(
        provider="BNL", log_dir=tmp_path, now=now, shard_index=2, num_shards=4
    )
    assert path == tmp_path / "2026-04-23" / "BNL-shard-2-of-4.log"


def test_configure_logging_threads_shard_kwargs(tmp_path: Path) -> None:
    """`configure_logging(...)` forwards shard kwargs through to the resolver."""
    path = configure_logging(
        provider="BNL", log_dir=tmp_path, shard_index=1, num_shards=3
    )
    assert path.name == "BNL-shard-1-of-3.log"


def test_configure_logging_disables_transformers_default_handler(tmp_path: Path) -> None:
    """transformers' built-in stderr handler must be disabled + set to propagate,
    otherwise its `warning` calls bypass our file + tqdm handlers."""
    from transformers.utils import logging as hf_logging

    with (
        patch.object(hf_logging, "disable_default_handler") as mock_disable,
        patch.object(hf_logging, "enable_propagation") as mock_propagate,
    ):
        configure_logging(provider="BNL", log_dir=tmp_path)

    mock_disable.assert_called_once()
    mock_propagate.assert_called_once()
