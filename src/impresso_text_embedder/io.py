"""S3 I/O helpers for impresso-text-embedder.

Thin wrappers around boto3 for the streaming reader, existence check, and
upload. See ``.history/io-layer/notes.md`` for the reasoning (including why
the three S3 helpers below are vendored rather than imported from
``impresso_essentials.io.s3``).
"""

from __future__ import annotations

import bz2
import hashlib
import logging
import os
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import boto3
from boto3.resources.base import ServiceResource
from boto3.s3.transfer import TransferConfig
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

DEFAULT_S3_HOST_URL = "https://os.zhdk.cloud.switch.ch/"

# Disable boto3's default flex-checksums for PutObject / GetObject. The
# defaults (``when_supported``) make PutObject send the body as ``aws-chunked``
# with a trailer and no Content-Length, which Ceph RadosGW (Switch Engines)
# rejects with ``MissingContentLength``. ``when_required`` restores a plain
# Content-Length'd PUT. Needs boto3>=1.36.5 / s3transfer>=0.11.2 to propagate
# through the high-level upload_file path. See
# ``.history/upload-integrity/notes.md``.
_S3_CONFIG = Config(
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)


def get_s3_client(host_url: str | None = None) -> BaseClient:
    """Return a boto3 S3 client authenticated via ``SE_*`` env vars.

    Reads ``SE_ACCESS_KEY``, ``SE_SECRET_KEY`` from the environment (the CLI
    entry points load ``.env`` once at startup). ``host_url`` falls back to
    ``SE_HOST_URL`` then to the Impresso default endpoint.
    """
    endpoint = host_url or os.environ.get("SE_HOST_URL") or DEFAULT_S3_HOST_URL
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["SE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["SE_SECRET_KEY"],
        endpoint_url=endpoint,
        config=_S3_CONFIG,
    )


def get_s3_resource(host_url: str | None = None) -> ServiceResource:
    """Return a boto3 S3 resource authenticated via ``SE_*`` env vars."""
    endpoint = host_url or os.environ.get("SE_HOST_URL") or DEFAULT_S3_HOST_URL
    return boto3.resource(
        "s3",
        aws_access_key_id=os.environ["SE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["SE_SECRET_KEY"],
        endpoint_url=endpoint,
        config=_S3_CONFIG,
    )

INPUT_FILENAME_RE = re.compile(r"^(?P<alias>[^/]+?)-(?P<year>\d{4})\.jsonl\.bz2$")
OUTPUT_PREFIX = "embeddings/docs"


class InputKey(NamedTuple):
    provider: str
    alias: str
    year: int
    key: str
    last_modified: datetime | None = None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into ``(bucket, key)``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    rest = uri[len("s3://") :]
    if "/" not in rest:
        raise ValueError(f"s3 URI missing key component: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"empty bucket or key in {uri!r}")
    return bucket, key


def build_output_key(provider: str, alias: str, year: int, model_slug: str) -> str:
    """Build the output key under the convention from CLAUDE.md.

    ``embeddings/docs/<model-slug>/<provider>/<alias>/<alias>-<year>.jsonl.bz2``
    """
    return f"{OUTPUT_PREFIX}/{model_slug}/{provider}/{alias}/{alias}-{year}.jsonl.bz2"


def parse_input_key(key: str, input_prefix: str = "") -> InputKey:
    """Reverse-map an input key to ``(provider, alias, year)``.

    ``key`` is the S3 key *without* the bucket. ``input_prefix`` is optional and
    stripped if present. Expects ``<prefix?>/<provider>/<alias>/<alias>-<year>.jsonl.bz2``.
    """
    trimmed = key
    if input_prefix:
        prefix = input_prefix.rstrip("/") + "/"
        if not trimmed.startswith(prefix):
            raise ValueError(f"key {key!r} does not start with prefix {input_prefix!r}")
        trimmed = trimmed[len(prefix) :]

    parts = trimmed.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"expected <provider>/<alias>/<alias>-<year>.jsonl.bz2 under prefix, got {trimmed!r}"
        )
    provider, alias, filename = parts
    m = INPUT_FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"filename {filename!r} does not match <alias>-<year>.jsonl.bz2")
    if m.group("alias") != alias:
        raise ValueError(
            f"filename alias {m.group('alias')!r} disagrees with directory alias {alias!r}"
        )
    return InputKey(provider=provider, alias=alias, year=int(m.group("year")), key=key)


def list_input_keys(
    bucket: str,
    provider: str,
    input_prefix: str = "",
    alias_filter: set[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> Iterator[InputKey]:
    """List ``.jsonl.bz2`` keys under ``<input_prefix>/<provider>/`` in ``bucket``.

    Uses the S3 client's ``list_objects_v2`` paginator and filters by filename shape.
    """
    s3 = get_s3_client()
    prefix_parts = [p for p in [input_prefix.strip("/"), provider] if p]
    prefix = "/".join(prefix_parts) + "/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith(".jsonl.bz2"):
                continue
            try:
                parsed = parse_input_key(key, input_prefix=input_prefix)
            except ValueError as exc:
                log.debug("skipping unrecognized key %s: %s", key, exc)
                continue
            if alias_filter is not None and parsed.alias not in alias_filter:
                continue
            if year_min is not None and parsed.year < year_min:
                continue
            if year_max is not None and parsed.year > year_max:
                continue
            yield parsed._replace(last_modified=obj.get("LastModified"))


def head_last_modified(bucket: str, key: str) -> datetime | None:
    """Return the object's ``LastModified`` (tz-aware UTC), or None if it doesn't exist."""
    s3 = get_s3_client()
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return resp.get("LastModified")


def iter_jsonl_bz2(bucket: str, key: str) -> Iterator[str]:
    """Stream lines from a ``.jsonl.bz2`` S3 object without loading it all in memory.

    Each yielded string is a decoded line without its trailing newline. Empty lines
    are skipped (they appear at the very end of some files).
    """
    s3r = get_s3_resource()
    body = s3r.Object(bucket, key).get()["Body"]
    with bz2.open(body, mode="rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line:
                yield line


def iter_jsonl_bz2_path(path: str | Path) -> Iterator[str]:
    """Stream lines from a local ``.jsonl.bz2`` file.

    Same semantics as :func:`iter_jsonl_bz2` but reads from disk. Used by the
    prefetched-download path in :mod:`pipeline`.
    """
    with bz2.open(str(path), mode="rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line:
                yield line


# Defaults chosen from boto3's S3 guide plus the rationale in
# ``.history/io-throughput/notes.md``. 8 MB chunks × 10 threads gives us
# parallel ranged GETs against Ceph RadosGW (Switch Engines) without paying the
# overhead of tiny parts on the long tail of small shards.
DEFAULT_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=10,
    use_threads=True,
)


def download_to_local(
    bucket: str,
    key: str,
    dest: str | Path,
    transfer_config: TransferConfig | None = None,
) -> None:
    """Download ``s3://bucket/key`` to a local path with multipart ranged GETs.

    Uses :class:`boto3.s3.transfer.TransferConfig` so files over the threshold
    are fetched concurrently. Raises on any failure (boto3's
    ``Bucket.download_file`` raises on non-2xx responses by default).
    """
    s3r = get_s3_resource()
    cfg = transfer_config if transfer_config is not None else DEFAULT_TRANSFER_CONFIG
    s3r.Bucket(bucket).download_file(key, str(dest), Config=cfg)
    log.debug("downloaded s3://%s/%s to %s", bucket, key, dest)


def _md5_file(path: str | Path, block: int = 8 * 1024 * 1024) -> str:
    """Stream-compute the lower-case hex MD5 of a local file.

    ``usedforsecurity=False`` is required on FIPS-enabled RHEL / NGC images:
    MD5 is used here only for S3 single-part ETag comparison, not security.
    """
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_uploaded_object(
    bucket: str, key: str, local_size: int, local_md5: str
) -> tuple[str, bool]:
    """HEAD ``s3://bucket/key`` and verify size + (single-part) ETag match.

    Returns ``(etag, is_multipart)``. Raises ``RuntimeError`` on mismatch;
    callers are expected to delete the bad object. Multipart ETags (``-N``
    suffix) are not reconstructed — size match is the floor.
    """
    s3 = get_s3_client()
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        raise RuntimeError(
            f"post-upload HEAD of s3://{bucket}/{key} failed: {exc}"
        ) from exc

    remote_size = resp["ContentLength"]
    if remote_size != local_size:
        raise RuntimeError(
            f"uploaded size mismatch for s3://{bucket}/{key}: "
            f"local={local_size} remote={remote_size}"
        )

    etag = resp["ETag"].strip('"')
    is_multipart = "-" in etag
    if is_multipart:
        log.debug(
            "multipart ETag %s for s3://%s/%s; skipping bit-exact check",
            etag,
            bucket,
            key,
        )
    elif etag.lower() != local_md5:
        raise RuntimeError(
            f"uploaded ETag mismatch for s3://{bucket}/{key}: "
            f"local_md5={local_md5} remote_etag={etag}"
        )
    return etag, is_multipart


def _best_effort_delete(bucket: str, key: str) -> None:
    """Delete ``s3://bucket/key``; log and swallow any error.

    Used to clean up an object that failed post-upload verification, so the
    next run doesn't permanently skip via ``--skip-if-s3-exists``.
    """
    try:
        get_s3_client().delete_object(Bucket=bucket, Key=key)
    except Exception as exc:
        log.warning(
            "failed to delete corrupted upload s3://%s/%s: %s", bucket, key, exc
        )


def upload_local_file(local_path: str | Path, bucket: str, key: str) -> None:
    """Upload a local file to ``s3://bucket/key`` and verify integrity.

    Computes the local file's MD5, uploads via the transfer manager, then
    HEADs the resulting object to confirm size matches and — for single-part
    uploads — that the returned ETag equals the local MD5. On any mismatch
    the uploaded object is deleted (best-effort) and a ``RuntimeError`` is
    raised.
    """
    cleaned_key = key.removeprefix("s3://")
    local_size = os.path.getsize(local_path)
    local_md5 = _md5_file(local_path)

    s3r = get_s3_resource()
    try:
        s3r.Bucket(bucket).upload_file(str(local_path), cleaned_key)
    except Exception as exc:
        raise RuntimeError(
            f"upload to s3://{bucket}/{cleaned_key} failed: {exc}"
        ) from exc

    try:
        etag, is_multipart = _verify_uploaded_object(
            bucket, cleaned_key, local_size, local_md5
        )
    except Exception:
        _best_effort_delete(bucket, cleaned_key)
        raise

    log.info(
        "uploaded %s -> s3://%s/%s size=%d etag=%s multipart=%s",
        local_path,
        bucket,
        cleaned_key,
        local_size,
        etag,
        is_multipart,
    )


def copy_s3_object(
    src_bucket: str,
    src_key: str,
    dst_bucket: str,
    dst_key: str,
    *,
    overwrite: bool = False,
) -> bool:
    """Server-side copy ``s3://src_bucket/src_key`` to ``s3://dst_bucket/dst_key``.

    Uses ``s3.copy_object``, which the storage backend executes without
    streaming the body through the client — same Ceph bucket means a
    pure metadata move. Idempotent: returns ``False`` without copying
    when the destination already exists and ``overwrite=False``.

    On any successful copy the destination is HEAD-checked and asserted
    against the source's ``ContentLength`` so a partial server-side
    copy doesn't silently land. Returns ``True`` when a copy happened.
    """
    s3 = get_s3_client()
    if not overwrite and head_last_modified(dst_bucket, dst_key) is not None:
        log.info(
            "copy skipped: s3://%s/%s already exists (use overwrite=True to replace)",
            dst_bucket,
            dst_key,
        )
        return False

    try:
        src_head = s3.head_object(Bucket=src_bucket, Key=src_key)
    except ClientError as exc:
        raise RuntimeError(
            f"source object s3://{src_bucket}/{src_key} not found: {exc}"
        ) from exc
    src_size = src_head["ContentLength"]

    try:
        s3.copy_object(
            CopySource={"Bucket": src_bucket, "Key": src_key},
            Bucket=dst_bucket,
            Key=dst_key,
        )
    except ClientError as exc:
        raise RuntimeError(
            f"copy_object s3://{src_bucket}/{src_key} -> "
            f"s3://{dst_bucket}/{dst_key} failed: {exc}"
        ) from exc

    dst_head = s3.head_object(Bucket=dst_bucket, Key=dst_key)
    dst_size = dst_head["ContentLength"]
    if dst_size != src_size:
        raise RuntimeError(
            f"post-copy size mismatch: src=s3://{src_bucket}/{src_key} ({src_size}) "
            f"vs dst=s3://{dst_bucket}/{dst_key} ({dst_size})"
        )

    log.info(
        "copied s3://%s/%s -> s3://%s/%s size=%d",
        src_bucket,
        src_key,
        dst_bucket,
        dst_key,
        dst_size,
    )
    return True
