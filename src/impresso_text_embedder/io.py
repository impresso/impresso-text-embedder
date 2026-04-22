"""S3 I/O helpers for impresso-text-embedder.

Thin layer over ``impresso_essentials.io.s3`` where its semantics match our needs,
and direct boto3 calls for the streaming reader and existence check. See
``.progress/io-layer/notes.md`` for the reasoning.
"""

from __future__ import annotations

import bz2
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from botocore.exceptions import ClientError
from impresso_essentials.io.s3 import (
    get_s3_client,
    get_s3_resource,
    upload_to_s3,
)

log = logging.getLogger(__name__)

INPUT_FILENAME_RE = re.compile(r"^(?P<alias>[^/]+?)-(?P<year>\d{4})\.jsonl\.bz2$")
OUTPUT_PREFIX = "embeddings/docs"


class InputKey(NamedTuple):
    provider: str
    alias: str
    year: int
    key: str


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
            yield parsed


def object_exists(bucket: str, key: str) -> bool:
    """Return True iff the S3 object exists. 404 → False; other errors re-raised."""
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


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


def upload_local_file(local_path: str | Path, bucket: str, key: str) -> None:
    """Upload a local file to ``s3://bucket/key``. Raises on failure."""
    ok = upload_to_s3(str(local_path), key, bucket)
    if not ok:
        raise RuntimeError(f"upload to s3://{bucket}/{key} failed (see logs)")
