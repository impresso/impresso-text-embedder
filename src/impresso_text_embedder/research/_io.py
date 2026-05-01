"""Context managers for the research-pipeline S3 plumbing.

The four research CLIs (``corpus_select``, ``corpus_fetch``,
``embed_sweep``, ``query_generate``) all need the same dance:
download an S3 input to a tempfile, write an output, optionally
upload it. This module collapses that boilerplate into two context
managers so each CLI's ``main()`` reduces to "config in, two ``with``
blocks, work in between".

Design:

- :func:`staged_input` always downloads to a tempfile (no
  "use-local-if-present" caching). Reasoning: a study YAML edit
  changes the S3 key, but a stale local mirror would silently
  override it. Always-download is the conservative default; if you
  want to re-iterate without re-downloading, copy the file out by
  hand before editing.
- :func:`staged_output` writes to a tempfile when uploading and to
  the study's local mirror when ``--no-upload`` is set. Tempfile
  on the upload path means a failed run does not pollute the mirror;
  the mirror path on the no-upload path means the user can find the
  artefact deterministically (``study_cfg.local_path(...)``).

The S3-path CLI flags (``--corpus-bucket``, ``--corpus-key``,
``--output-bucket``, ``--output-prefix``, ``--output-key``,
``--local-input``, ``--local-output``) are intentionally absent
from the CLIs that consume these helpers — the study config is the
single source of truth for "where things live", and ``--no-upload``
is the only iteration knob that survives.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from impresso_text_embedder import io as s3io

log = logging.getLogger(__name__)


@contextmanager
def staged_input(bucket: str, s3_key: str) -> Iterator[Path]:
    """Yield a local :class:`Path` pointing at a freshly downloaded copy
    of ``s3://bucket/s3_key``.

    The tempfile is removed on context exit, even on exceptions.
    Download failures bubble up as :class:`RuntimeError` /
    :class:`ClientError` from the underlying boto3 helper.
    """
    with tempfile.NamedTemporaryFile(
        prefix="staged-in-", suffix=".jsonl.bz2", delete=False
    ) as tmp:
        local = Path(tmp.name)
    log.info("downloading s3://%s/%s -> %s", bucket, s3_key, local)
    try:
        s3io.download_to_local(bucket, s3_key, local)
        yield local
    finally:
        try:
            local.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def staged_output(
    bucket: str,
    s3_key: str,
    local_mirror: Path,
    *,
    upload: bool = True,
) -> Iterator[Path]:
    """Yield a local :class:`Path` to write to.

    On context exit:

    - ``upload=True``: upload the file to ``s3://bucket/s3_key`` and
      remove the tempfile. The upload uses
      :func:`io.upload_local_file` which verifies size + ETag and
      best-effort-deletes corrupted uploads.
    - ``upload=False``: leave the file at ``local_mirror`` (no upload,
      no cleanup). The mirror's parent directory is created on entry.

    The split between tempfile (upload path) and mirror (no-upload
    path) is deliberate: a failed upload run does not pollute the
    mirror, and a successful no-upload run lands at a deterministic
    path the user can find without scrolling the log.

    Exceptions raised inside the ``with`` block always abort the
    upload — the body must finish cleanly for the artefact to ship.
    """
    if upload:
        with tempfile.NamedTemporaryFile(
            prefix="staged-out-", suffix=".jsonl.bz2", delete=False
        ) as tmp:
            local = Path(tmp.name)
        try:
            yield local
            s3io.upload_local_file(local, bucket, s3_key)
            log.info("uploaded %s -> s3://%s/%s", local, bucket, s3_key)
        finally:
            try:
                local.unlink()
            except FileNotFoundError:
                pass
    else:
        local_mirror.parent.mkdir(parents=True, exist_ok=True)
        yield local_mirror
        log.info("wrote %s (no upload)", local_mirror)


__all__ = ["staged_input", "staged_output"]
