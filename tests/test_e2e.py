"""End-to-end: create embeddings for a fake provider, then validate them.

No S3, no real model. Exercises both CLI entry points through ``main()`` and
asserts the produced output is self-consistent (passes structural + self-comparison
checks via the validate CLI).
"""

from __future__ import annotations

import bz2
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from impresso_text_embedder import embed as em
from impresso_text_embedder import io as s3io
from impresso_text_embedder import model as model_mod
from impresso_text_embedder import pipeline as pl
from impresso_text_embedder.cli import create as create_cli
from impresso_text_embedder.cli import validate as validate_cli


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _record(ci_id: str, body: str) -> dict:
    return {
        "id": ci_id,
        "tp": "ar",
        "lg": "fr",
        "sents": [{"tok": [{"t": body, "o": 0}]}],
    }


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    """Set up a fake S3: input files on disk, upload captures to disk."""
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    input_root.mkdir()
    output_root.mkdir()

    # Two input files under one provider/alias, two different years.
    for year in (1910, 1911):
        path = input_root / f"SNL/EXP/EXP-{year}.jsonl.bz2"
        path.parent.mkdir(parents=True, exist_ok=True)
        with bz2.open(path, "wt", encoding="utf-8") as fh:
            for i in range(3):
                fh.write(
                    json.dumps(
                        _record(
                            f"ci-{year}-{i}",
                            f"Article body for {year} item {i}, long enough to pass the min length filter.",
                        )
                    )
                    + "\n"
                )

    def fake_iter_jsonl_bz2(bucket, key):
        path = input_root / key
        with bz2.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line

    def fake_list_input_keys(bucket, provider, input_prefix="", **_):
        # Walk the fake input root and return InputKey objects with real mtimes.
        provider_dir = input_root / provider
        if not provider_dir.exists():
            return
        for path in sorted(provider_dir.rglob("*.jsonl.bz2")):
            rel = path.relative_to(input_root).as_posix()
            parsed = s3io.parse_input_key(rel)
            yield parsed._replace(last_modified=_mtime(path))

    uploaded: dict[str, bytes] = {}

    def fake_upload(local_path, bucket, key):
        uploaded[key] = Path(local_path).read_bytes()
        # Mirror into output_root for later reading.
        dest = output_root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(uploaded[key])

    def fake_head_last_modified(bucket, key):
        dest = output_root / key
        if not dest.exists():
            return None
        return _mtime(dest)

    monkeypatch.setattr(s3io, "iter_jsonl_bz2", fake_iter_jsonl_bz2)
    monkeypatch.setattr(s3io, "list_input_keys", fake_list_input_keys)
    monkeypatch.setattr(s3io, "upload_local_file", fake_upload)
    monkeypatch.setattr(s3io, "head_last_modified", fake_head_last_modified)
    # pipeline.iter_input_lines goes through s3io.iter_jsonl_bz2 already.

    # Deterministic fake encoder: derive embedding from text length modulo.
    def fake_encode(model, texts, **_):
        return np.asarray(
            [[float(len(t) % 7), float(len(t) % 11), 1.0] for t in texts],
            dtype=np.float32,
        )

    monkeypatch.setattr(em, "encode_texts", fake_encode)

    # Avoid downloading the real model; return a MagicMock that the mocked encoder ignores.
    monkeypatch.setattr(model_mod, "load_model", lambda **_: MagicMock())

    return {
        "input_root": input_root,
        "output_root": output_root,
        "uploaded": uploaded,
    }


def test_create_then_validate_roundtrip(fake_world, capsys):
    # Create
    rc = create_cli.main(
        [
            "--provider",
            "SNL",
            "--input-bucket",
            "in-bkt",
            "--output-bucket",
            "out-bkt",
            "--embedding-level",
            "text",
            "--batch-size",
            "4",
            "--min-char-length",
            "10",
        ]
    )
    assert rc == 0

    uploaded = fake_world["uploaded"]
    assert set(uploaded) == {
        "embeddings/docs/gte-multilingual-base/SNL/EXP/EXP-1910.jsonl.bz2",
        "embeddings/docs/gte-multilingual-base/SNL/EXP/EXP-1911.jsonl.bz2",
    }

    # Inspect one output file.
    key = "embeddings/docs/gte-multilingual-base/SNL/EXP/EXP-1910.jsonl.bz2"
    lines = bz2.decompress(uploaded[key]).decode("utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["id"] for r in records] == ["ci-1910-0", "ci-1910-1", "ci-1910-2"]
    for r in records:
        assert r["embedder"] == "Alibaba-NLP/gte-multilingual-base@default"
        assert isinstance(r["embedding"], list) and len(r["embedding"]) == 3

    # Validate (structural + self-comparison) via the validate CLI.
    out_path = fake_world["output_root"] / key
    capsys.readouterr()  # clear previous output
    rc = validate_cli.main([str(out_path)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out

    rc = validate_cli.main([str(out_path), "--target", str(out_path)])
    assert rc == 0


def test_second_run_skips_existing_outputs(fake_world):
    argv = [
        "--provider",
        "SNL",
        "--input-bucket",
        "in-bkt",
        "--output-bucket",
        "out-bkt",
        "--embedding-level",
        "text",
        "--batch-size",
        "4",
        "--min-char-length",
        "10",
    ]
    assert create_cli.main(argv) == 0
    # Second run should produce no new uploads because outputs exist.
    before = dict(fake_world["uploaded"])

    def explode(*a, **kw):
        raise AssertionError("upload should not be called on second run")

    with patch.object(s3io, "upload_local_file", explode):
        assert create_cli.main(argv) == 0

    assert fake_world["uploaded"] == before


def test_force_flag_reprocesses(fake_world):
    argv = [
        "--provider",
        "SNL",
        "--input-bucket",
        "in-bkt",
        "--output-bucket",
        "out-bkt",
        "--embedding-level",
        "text",
        "--batch-size",
        "4",
        "--min-char-length",
        "10",
    ]
    assert create_cli.main(argv) == 0
    seen_before = len(fake_world["uploaded"])
    # Re-run with --force; uploads happen again (overwriting).
    assert create_cli.main([*argv, "--force"]) == 0
    # Same set of keys, count unchanged (we overwrite in fake).
    assert len(fake_world["uploaded"]) == seen_before


def test_reembeds_when_input_is_newer(fake_world):
    argv = [
        "--provider",
        "SNL",
        "--input-bucket",
        "in-bkt",
        "--output-bucket",
        "out-bkt",
        "--embedding-level",
        "text",
        "--batch-size",
        "4",
        "--min-char-length",
        "10",
    ]
    assert create_cli.main(argv) == 0
    first_run_keys = set(fake_world["uploaded"])
    assert first_run_keys  # sanity

    # Touch one input file so its mtime is strictly newer than the output's.
    input_root: Path = fake_world["input_root"]
    output_root: Path = fake_world["output_root"]
    target_input = input_root / "SNL/EXP/EXP-1910.jsonl.bz2"
    target_output = (
        output_root / "embeddings/docs/gte-multilingual-base/SNL/EXP/EXP-1910.jsonl.bz2"
    )
    new_mtime = target_output.stat().st_mtime + 10
    os.utime(target_input, (new_mtime, new_mtime))

    reuploaded: list[str] = []
    orig_upload = s3io.upload_local_file

    def spy_upload(local_path, bucket, key):
        reuploaded.append(key)
        orig_upload(local_path, bucket, key)

    with patch.object(s3io, "upload_local_file", spy_upload):
        assert create_cli.main(argv) == 0

    # Only the touched file should have been re-embedded.
    assert reuploaded == [
        "embeddings/docs/gte-multilingual-base/SNL/EXP/EXP-1910.jsonl.bz2"
    ]


def test_dry_run_does_no_work(fake_world):
    argv = [
        "--provider",
        "SNL",
        "--input-bucket",
        "in-bkt",
        "--output-bucket",
        "out-bkt",
        "--dry-run",
    ]
    # No uploads; process_file should not run.
    with patch.object(pl, "process_file", side_effect=AssertionError("no work in dry run")):
        assert create_cli.main(argv) == 0
    assert fake_world["uploaded"] == {}
