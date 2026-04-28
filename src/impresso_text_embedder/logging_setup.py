"""Two-handler logging for ``impresso-embed-create``.

Wires the root logger with:

* a ``FileHandler`` writing full detail (level ``--log-level-file``,
  default ``INFO``) to
  ``/rcp-scratch/<username>/experiments/embeddings/<YYYY-MM-DD>/<provider>.log``
  (or ``<log-dir>/<YYYY-MM-DD>/<provider>.log`` when ``--log-dir`` is
  given);
* a :class:`TqdmLoggingHandler` at level ``ERROR`` that routes records
  through :meth:`tqdm.tqdm.write` so stack traces land above the progress
  bar without breaking it.

If ``/rcp-scratch`` is not mounted and no override is passed, the CLI
exits non-zero with a clear message — no silent fallback. Rationale in
``.progress/structured-logging/notes.md``.
"""

from __future__ import annotations

import getpass
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

DEFAULT_RCP_SCRATCH = Path("/rcp-scratch")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class TqdmLoggingHandler(logging.Handler):
    """Route log records through ``tqdm.write`` so a live bar stays intact."""

    def __init__(self, level: int = logging.ERROR) -> None:
        super().__init__(level=level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def _resolve_log_path(
    provider: str,
    log_dir: Path | None,
    now: datetime | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> Path:
    """Return the full log-file path; fail fast if the default base is missing.

    When ``log_dir`` is ``None``, the base is
    ``/rcp-scratch/<getpass.getuser()>/experiments/embeddings`` and the base
    must already exist (``/rcp-scratch`` specifically — PVC not mounted is
    the common failure mode). Otherwise ``log_dir`` is used verbatim as the
    base; its parents are created if missing.

    The date component is ``YYYY-MM-DD`` (UTC). Filename is ``<provider>.log``
    by default; when ``num_shards`` is greater than 1 it becomes
    ``<provider>-shard-<i>-of-<N>.log`` so concurrent shards don't overwrite
    each other's logs (step 18, multi-gpu-sharding).
    """
    date_str = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    if num_shards is not None and num_shards > 1:
        filename = f"{provider}-shard-{shard_index}-of-{num_shards}.log"
    else:
        filename = f"{provider}.log"

    if log_dir is None:
        if not DEFAULT_RCP_SCRATCH.is_dir():
            raise SystemExit(
                f"Default log path requires {DEFAULT_RCP_SCRATCH} to be mounted "
                "(Run:AI PVC). Mount it, or pass --log-dir <path> to override."
            )
        base = DEFAULT_RCP_SCRATCH / getpass.getuser() / "experiments" / "embeddings"
    else:
        base = Path(log_dir)

    path = base / date_str / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _route_third_party_through_root() -> None:
    """Force noisy libraries to propagate through our root handlers.

    ``transformers`` (and transitively ``sentence_transformers``) installs
    its own stderr handler on import, which bypasses the file + tqdm
    handlers we just wired. Disable that handler and enable propagation so
    its records flow through the root logger instead. Leaves levels alone
    — our root handlers already gate what reaches the terminal (ERROR) vs.
    the file (``--log-level-file``).
    """
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except Exception:  # pragma: no cover - transformers missing in some envs
        pass


def configure_logging(
    provider: str,
    log_dir: Path | None = None,
    log_level_file: str = "INFO",
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> Path:
    """Install the file + tqdm-ERROR handlers on the root logger.

    Returns the resolved log-file path so the caller can print it on
    startup.
    """
    path = _resolve_log_path(
        provider, log_dir, shard_index=shard_index, num_shards=num_shards
    )

    root = logging.getLogger()
    # Drop any prior handlers so re-invoking (e.g. in tests) is clean.
    for h in list(root.handlers):
        root.removeHandler(h)

    file_level = getattr(logging, log_level_file.upper(), logging.INFO)
    terminal_level = logging.ERROR
    # Root must be at or below the most permissive handler level, or records
    # are dropped before they reach the handlers.
    root.setLevel(min(file_level, terminal_level))

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    tqdm_handler = TqdmLoggingHandler(level=terminal_level)
    tqdm_handler.setFormatter(formatter)
    root.addHandler(tqdm_handler)

    _route_third_party_through_root()

    return path


__all__ = ["TqdmLoggingHandler", "configure_logging"]
