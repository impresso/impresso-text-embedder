from __future__ import annotations

import bz2
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from impresso_text_embedder import embed as em
from impresso_text_embedder import io as s3io
from impresso_text_embedder import pipeline as pl

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_T_LATER = _T0 + timedelta(hours=1)


def _cfg(force=False, level="text"):
    return pl.PipelineConfig(
        input_bucket="in-bkt",
        output_bucket="out-bkt",
        input_prefix="",
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_revision=None,
        level=level,
        chunking_strategy_name="semantic",
        force=force,
        encoder=em.EncoderConfig(
            batch_size=2,
            normalize_embeddings=False,
            min_char_length=5,
            include_text=False,
            content_types=frozenset({"ar"}),
        ),
    )


def test_model_slug_strips_vendor():
    assert pl.model_slug("Alibaba-NLP/gte-multilingual-base") == "gte-multilingual-base"
    assert pl.model_slug("BAAI/bge-m3") == "bge-m3"
    assert pl.model_slug("local-name") == "local-name"


def test_iter_parsed_skips_malformed(caplog):
    lines = ['{"id": "a"}', "not json", '{"id": "b"}']
    with caplog.at_level("WARNING"):
        out = list(pl._iter_parsed(lines))
    assert [r["id"] for r in out] == ["a", "b"]
    assert any("malformed JSON" in r.message for r in caplog.records)


@pytest.fixture
def fake_input_records(monkeypatch):
    """Three records: one short (skipped), two long enough."""
    records = [
        {
            "id": "keep-1",
            "tp": "ar",
            "sents": [{"tok": [{"t": "This article body is long enough to pass.", "o": 0}]}],
        },
        {"id": "skip-short", "tp": "ar", "sents": [{"tok": [{"t": "tiny", "o": 0}]}]},
        {
            "id": "keep-2",
            "tp": "ar",
            "sents": [{"tok": [{"t": "Another sufficiently long article body.", "o": 0}]}],
        },
    ]

    def fake_iter(bucket, key):
        for r in records:
            yield json.dumps(r)

    monkeypatch.setattr(pl, "iter_input_lines", fake_iter)
    return records


def test_process_file_writes_and_uploads(monkeypatch, fake_input_records, tmp_path):
    uploaded = {}

    def fake_upload(local_path, bucket, key):
        uploaded["bucket"] = bucket
        uploaded["key"] = key
        uploaded["bytes"] = Path(local_path).read_bytes()

    monkeypatch.setattr(s3io, "upload_local_file", fake_upload)
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.1, 0.2] for _ in texts], dtype=np.float32),
    )

    input_key = s3io.InputKey("SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2")
    cfg = _cfg()
    assert pl.process_file(input_key, MagicMock(), cfg) is True
    assert uploaded["bucket"] == "out-bkt"
    assert (
        uploaded["key"]
        == "embeddings/docs/gte-multilingual-base/SNL/EXP/EXP-1912.jsonl.bz2"
    )
    body = bz2.decompress(uploaded["bytes"]).decode("utf-8").splitlines()
    records = [json.loads(line) for line in body]
    assert [r["id"] for r in records] == ["keep-1", "keep-2"]
    for r in records:
        assert r["embedder"] == "Alibaba-NLP/gte-multilingual-base@default"
        assert r["embedding"] == [0.1, 0.2]


def test_process_file_skips_when_output_exists_and_no_input_timestamp(monkeypatch):
    # InputKey without last_modified → fallback "output exists ⇒ skip" semantics.
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: _T0)
    monkeypatch.setattr(
        s3io,
        "upload_local_file",
        MagicMock(side_effect=AssertionError("upload must not be called on skip")),
    )
    input_key = s3io.InputKey("SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2")
    assert pl.process_file(input_key, MagicMock(), _cfg(force=False)) is False


def test_process_file_skips_when_output_newer_than_input(monkeypatch):
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: _T_LATER)
    monkeypatch.setattr(
        s3io,
        "upload_local_file",
        MagicMock(side_effect=AssertionError("upload must not be called on skip")),
    )
    input_key = s3io.InputKey("SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2", last_modified=_T0)
    assert pl.process_file(input_key, MagicMock(), _cfg(force=False)) is False


def test_process_file_reprocesses_when_input_newer_than_output(
    monkeypatch, fake_input_records, caplog
):
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: _T0)
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.0, 0.0] for _ in texts], dtype=np.float32),
    )
    input_key = s3io.InputKey(
        "SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2", last_modified=_T_LATER
    )
    with caplog.at_level("INFO"):
        assert pl.process_file(input_key, MagicMock(), _cfg(force=False)) is True
    assert any("input newer than output" in r.message for r in caplog.records)


def test_process_file_force_overrides_skip(monkeypatch, fake_input_records):
    # --force must not even consult head_last_modified.
    monkeypatch.setattr(
        s3io,
        "head_last_modified",
        MagicMock(side_effect=AssertionError("head must not be called under --force")),
    )
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.0, 0.0] for _ in texts], dtype=np.float32),
    )
    input_key = s3io.InputKey(
        "SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2", last_modified=_T0
    )
    assert pl.process_file(input_key, MagicMock(), _cfg(force=True)) is True


def test_process_provider_dry_run(monkeypatch):
    keys = [
        s3io.InputKey("SNL", "EXP", 1910, "SNL/EXP/EXP-1910.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1911, "SNL/EXP/EXP-1911.jsonl.bz2", last_modified=_T0),
    ]
    monkeypatch.setattr(s3io, "list_input_keys", lambda **kw: iter(keys))
    # 1910 output is newer than its input → skip. 1911 output is missing → would process.
    monkeypatch.setattr(
        s3io,
        "head_last_modified",
        lambda b, k: _T_LATER if k.endswith("EXP-1910.jsonl.bz2") else None,
    )
    monkeypatch.setattr(pl, "process_file", MagicMock(side_effect=AssertionError("no work in dry run")))

    summary = pl.process_provider("SNL", model=None, cfg=_cfg(), dry_run=True)
    assert summary["processed"] == 1
    assert summary["skipped"] == 1
    assert len(summary["files"]) == 2


def test_with_batch_size_updates_encoder_only():
    base = _cfg()
    updated = pl.with_batch_size(base, 128)
    assert updated.encoder.batch_size == 128
    assert base.encoder.batch_size == 2  # original unchanged
    assert updated.input_bucket == base.input_bucket
