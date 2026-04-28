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
            min_char_length=5,
            content_types=frozenset({"ar"}),
        ),
    )


def test_model_slug_uses_impresso_override_then_strips_vendor():
    # Shipped model uses the Impresso-convention slug.
    assert pl.model_slug("Alibaba-NLP/gte-multilingual-base") == "embeddings_gte_v1-1-0"
    # Unknown vendor-prefixed name falls back to stripping the vendor.
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
    """Three records: one short (skipped), two long enough.

    The prefetcher downloads real ``.jsonl.bz2`` files, so the fake materialises
    the records to ``dest`` whenever ``io.download_to_local`` is called.
    """
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

    def fake_download(bucket, key, dest, transfer_config=None):
        with bz2.open(dest, "wt", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    monkeypatch.setattr(s3io, "download_to_local", fake_download)
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
        == "embeddings/docs/embeddings_gte_v1-1-0/SNL/EXP/EXP-1912.jsonl.bz2"
    )
    body = bz2.decompress(uploaded["bytes"]).decode("utf-8").splitlines()
    records = [json.loads(line) for line in body]
    assert [r["ci_id"] for r in records] == ["keep-1", "keep-2"]
    for r in records:
        assert r["model_id"] == "Alibaba-NLP/gte-multilingual-base@default"
        assert r["embedding"] == [0.1, 0.2]
        assert r["size"] == len(r["embedding"])


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


def test_process_provider_overlaps_prefetch_and_upload(monkeypatch, tmp_path):
    """File N+1's download must start before file N's upload has completed.

    Asserts the pipeline actually overlaps stages rather than running them
    serially. Uses two threading.Events to gate the stages so we can observe
    their interleaving.
    """
    import threading

    keys = [
        s3io.InputKey("SNL", "EXP", 1910, "SNL/EXP/EXP-1910.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1911, "SNL/EXP/EXP-1911.jsonl.bz2", last_modified=_T0),
    ]
    monkeypatch.setattr(s3io, "list_input_keys", lambda **kw: iter(keys))
    # No existing outputs → both files will be processed.
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.1, 0.2] for _ in texts], dtype=np.float32),
    )

    downloads: list[str] = []
    upload_started = threading.Event()
    upload_can_finish = threading.Event()

    def fake_download(bucket, key, dest, transfer_config=None):
        downloads.append(key)
        # Materialise a one-record file so encode has something to do.
        rec = {
            "id": f"rec-{key}",
            "tp": "ar",
            "sents": [{"tok": [{"t": "long enough body to pass the min char filter", "o": 0}]}],
        }
        with bz2.open(dest, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    uploads: list[str] = []

    def fake_upload(local_path, bucket, key):
        uploads.append(key)
        upload_started.set()
        # Block so we can observe that the *next* prefetch started while this
        # upload was still running.
        assert upload_can_finish.wait(timeout=5.0), "upload never unblocked"

    monkeypatch.setattr(s3io, "download_to_local", fake_download)
    monkeypatch.setattr(s3io, "upload_local_file", fake_upload)

    # Run process_provider on a background thread so the main thread can
    # coordinate the upload barrier.
    result: dict = {}

    def run():
        result["summary"] = pl.process_provider("SNL", model=MagicMock(), cfg=_cfg())

    t = threading.Thread(target=run)
    t.start()
    assert upload_started.wait(timeout=5.0), "first upload never started"

    # At this point file 0's upload is blocked. If the pipeline overlaps
    # correctly, file 1's download must already have started.
    assert "SNL/EXP/EXP-1911.jsonl.bz2" in downloads, (
        f"prefetch did not overlap upload; downloads so far: {downloads}"
    )

    upload_can_finish.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert result["summary"]["processed"] == 2
    assert uploads == [
        "embeddings/docs/embeddings_gte_v1-1-0/SNL/EXP/EXP-1910.jsonl.bz2",
        "embeddings/docs/embeddings_gte_v1-1-0/SNL/EXP/EXP-1911.jsonl.bz2",
    ]


def test_process_provider_dry_run_respects_limit(monkeypatch):
    keys = [
        s3io.InputKey("SNL", "EXP", 1910, "SNL/EXP/EXP-1910.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1911, "SNL/EXP/EXP-1911.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2", last_modified=_T0),
    ]
    monkeypatch.setattr(s3io, "list_input_keys", lambda **kw: iter(keys))
    head_calls: list[str] = []

    def fake_head(bucket, key):
        head_calls.append(key)
        return None  # output missing → would process

    monkeypatch.setattr(s3io, "head_last_modified", fake_head)

    summary = pl.process_provider("SNL", model=None, cfg=_cfg(), limit=2, dry_run=True)

    assert summary["processed"] == 2
    assert summary["skipped"] == 0
    assert len(summary["files"]) == 2
    assert all("EXP-1912" not in k for k in head_calls), (
        f"head must not be called for the shard past the limit; got {head_calls}"
    )


def test_process_provider_limit_applies_before_skip(monkeypatch):
    """An already-processed shard still counts toward --limit."""
    keys = [
        s3io.InputKey("SNL", "EXP", 1910, "SNL/EXP/EXP-1910.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1911, "SNL/EXP/EXP-1911.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2", last_modified=_T0),
    ]
    monkeypatch.setattr(s3io, "list_input_keys", lambda **kw: iter(keys))
    # 1910's output is newer → skip. 1911's output is missing → process. 1912 must not be consulted.
    monkeypatch.setattr(
        s3io,
        "head_last_modified",
        lambda b, k: _T_LATER if k.endswith("EXP-1910.jsonl.bz2") else None,
    )

    summary = pl.process_provider("SNL", model=None, cfg=_cfg(), limit=2, dry_run=True)

    assert summary["processed"] == 1
    assert summary["skipped"] == 1
    assert len(summary["files"]) == 2
    assert not any(k.endswith("EXP-1912.jsonl.bz2") for k in summary["files"])


def test_process_provider_limit_zero(monkeypatch):
    keys = [
        s3io.InputKey("SNL", "EXP", 1910, "SNL/EXP/EXP-1910.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1911, "SNL/EXP/EXP-1911.jsonl.bz2", last_modified=_T0),
    ]
    monkeypatch.setattr(s3io, "list_input_keys", lambda **kw: iter(keys))
    monkeypatch.setattr(
        s3io,
        "head_last_modified",
        MagicMock(side_effect=AssertionError("head must not be called with limit=0")),
    )

    summary = pl.process_provider("SNL", model=None, cfg=_cfg(), limit=0, dry_run=True)

    assert summary == {"processed": 0, "skipped": 0, "files": []}


def test_process_provider_updates_tqdm_postfix_per_file(monkeypatch):
    """After each completed file, `pbar.set_postfix` is called with timing stats."""
    keys = [
        s3io.InputKey("SNL", "EXP", 1910, "SNL/EXP/EXP-1910.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1911, "SNL/EXP/EXP-1911.jsonl.bz2", last_modified=_T0),
    ]
    monkeypatch.setattr(s3io, "list_input_keys", lambda **kw: iter(keys))
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.1, 0.2] for _ in texts], dtype=np.float32),
    )

    def fake_download(bucket, key, dest, transfer_config=None):
        rec = {
            "id": f"rec-{key}",
            "tp": "ar",
            "sents": [{"tok": [{"t": "long enough body to pass the min char filter", "o": 0}]}],
        }
        with bz2.open(dest, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    monkeypatch.setattr(s3io, "download_to_local", fake_download)
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)

    # Patch the tqdm bar in pipeline.py with a spy that records set_postfix calls.
    postfix_calls: list[dict] = []

    class SpyBar:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def set_description(self, *_a, **_k):
            pass

        def set_postfix(self, **kw):
            postfix_calls.append(kw)

        def update(self, _n):
            pass

    monkeypatch.setattr(pl, "tqdm", SpyBar)

    summary = pl.process_provider("SNL", model=MagicMock(), cfg=_cfg())
    assert summary["processed"] == 2
    assert len(postfix_calls) == 2
    # Each postfix has at least encode timing (download / upload_wait may be near zero).
    for call in postfix_calls:
        assert "enc" in call


def test_process_provider_emits_progress_log_lines(monkeypatch, caplog):
    """Under non-tty (runai submit), progress is recoverable from the log file.

    Asserts the three log lines that make `tail -f` show position regardless
    of TTY: a one-shot ``planned: …`` denominator, a ``start i/N …`` per
    file, and the existing ``done`` line augmented with ``i/N``.
    """
    keys = [
        s3io.InputKey("SNL", "EXP", 1910, "SNL/EXP/EXP-1910.jsonl.bz2", last_modified=_T0),
        s3io.InputKey("SNL", "EXP", 1911, "SNL/EXP/EXP-1911.jsonl.bz2", last_modified=_T0),
    ]
    monkeypatch.setattr(s3io, "list_input_keys", lambda **kw: iter(keys))
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.1, 0.2] for _ in texts], dtype=np.float32),
    )

    def fake_download(bucket, key, dest, transfer_config=None):
        rec = {
            "id": f"rec-{key}",
            "tp": "ar",
            "sents": [{"tok": [{"t": "long enough body to pass the min char filter", "o": 0}]}],
        }
        with bz2.open(dest, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    monkeypatch.setattr(s3io, "download_to_local", fake_download)
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)

    with caplog.at_level("INFO", logger="impresso_text_embedder.pipeline"):
        summary = pl.process_provider("SNL", model=MagicMock(), cfg=_cfg())

    assert summary["processed"] == 2

    messages = [r.message for r in caplog.records]

    planned = [m for m in messages if m.startswith("planned:")]
    assert planned == ["planned: provider=SNL to_process=2 already_done=0"]

    starts = [m for m in messages if m.startswith("start ")]
    assert starts == [
        "start 1/2 EXP/1910 key=SNL/EXP/EXP-1910.jsonl.bz2",
        "start 2/2 EXP/1911 key=SNL/EXP/EXP-1911.jsonl.bz2",
    ]

    done = [m for m in messages if m.startswith("done ")]
    assert len(done) == 2
    assert done[0].startswith("done 1/2 s3://out-bkt/")
    assert done[1].startswith("done 2/2 s3://out-bkt/")
    # ETA appears on every line except the last (no remaining files to project).
    assert " eta=" in done[0] and " eta_at=" in done[0]
    assert " eta=" not in done[1] and " eta_at=" not in done[1]


def test_format_eta_renders_hours_minutes_seconds():
    assert pl._format_eta(0) == "0:00:00"
    assert pl._format_eta(59) == "0:00:59"
    assert pl._format_eta(60) == "0:01:00"
    assert pl._format_eta(3661) == "1:01:01"
    # Hours are uncapped — a multi-day backfill should still render cleanly.
    assert pl._format_eta(90061) == "25:01:01"
    # Negatives clamp to zero rather than rendering ``-0:00:01``.
    assert pl._format_eta(-5) == "0:00:00"


def test_with_batch_size_updates_encoder_only():
    base = _cfg()
    updated = pl.with_batch_size(base, 128)
    assert updated.encoder.batch_size == 128
    assert base.encoder.batch_size == 2  # original unchanged
    assert updated.input_bucket == base.input_bucket


def test_process_file_done_log_includes_filter_tally(
    monkeypatch, fake_input_records, caplog
):
    """Short records counted in the per-file ``skipped=`` clause of the done log."""
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.1, 0.2] for _ in texts], dtype=np.float32),
    )

    input_key = s3io.InputKey("SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2")
    with caplog.at_level("INFO", logger="impresso_text_embedder.pipeline"):
        assert pl.process_file(input_key, MagicMock(), _cfg()) is True

    done_lines = [r.message for r in caplog.records if r.message.startswith("done ")]
    assert done_lines, "expected a done INFO line"
    [done] = done_lines
    # fake_input_records has 1 short ("tiny") and 2 long records → skipped=1 too_short=1.
    assert "records=2" in done
    assert "skipped=1 (too_short=1)" in done


def test_process_file_done_log_omits_skipped_when_clean(
    monkeypatch, caplog, tmp_path
):
    """When nothing is filtered, the done line must not grow a ``skipped=`` clause."""
    import bz2 as _bz2

    def fake_download(bucket, key, dest, transfer_config=None):
        # One record that passes all filters.
        rec = {
            "id": "keep-1",
            "tp": "ar",
            "sents": [{"tok": [{"t": "long enough body to pass the min char filter", "o": 0}]}],
        }
        with _bz2.open(dest, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    monkeypatch.setattr(s3io, "download_to_local", fake_download)
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)
    monkeypatch.setattr(
        em,
        "encode_texts",
        lambda model, texts, **_: np.asarray([[0.1, 0.2] for _ in texts], dtype=np.float32),
    )

    input_key = s3io.InputKey("SNL", "EXP", 1912, "SNL/EXP/EXP-1912.jsonl.bz2")
    with caplog.at_level("INFO", logger="impresso_text_embedder.pipeline"):
        assert pl.process_file(input_key, MagicMock(), _cfg()) is True

    [done] = [r.message for r in caplog.records if r.message.startswith("done ")]
    assert "records=1" in done
    assert "skipped=" not in done


# --- step 18: multi-gpu file-list sharding -----------------------------------


def _mk_keys(n: int) -> list[s3io.InputKey]:
    """Build N synthetic InputKeys in lexicographic key order."""
    return [
        s3io.InputKey("SNL", "EXP", 1900 + i, f"SNL/EXP/EXP-{1900 + i}.jsonl.bz2")
        for i in range(n)
    ]


def test_apply_shard_filter_partition_is_disjoint_and_complete():
    """Round-robin over N=4: union covers all, intersections are empty."""
    keys = _mk_keys(23)
    parts = [list(pl._apply_shard_filter(iter(keys), i, 4)) for i in range(4)]
    # Each shard's keys agree with the modulo predicate.
    for i, part in enumerate(parts):
        for k in part:
            assert keys.index(k) % 4 == i
    # Union = original list (set comparison; order irrelevant).
    union = set().union(*[set(p) for p in parts])
    assert union == set(keys)
    # Pairwise disjoint.
    for i in range(4):
        for j in range(i + 1, 4):
            assert set(parts[i]).isdisjoint(parts[j])


def test_apply_shard_filter_round_robin_indices():
    """Shard 0 of 4 picks {0,4,8,…}, shard 1 picks {1,5,9,…}, etc."""
    keys = _mk_keys(12)
    assert list(pl._apply_shard_filter(iter(keys), 0, 4)) == [keys[0], keys[4], keys[8]]
    assert list(pl._apply_shard_filter(iter(keys), 1, 4)) == [keys[1], keys[5], keys[9]]
    assert list(pl._apply_shard_filter(iter(keys), 3, 4)) == [keys[3], keys[7], keys[11]]


def test_apply_shard_filter_default_passthrough():
    """num_shards=1 is a no-op — returns the original iterable identity."""
    keys = _mk_keys(5)
    out = pl._apply_shard_filter(keys, 0, 1)
    assert out is keys  # not even wrapped


def test_apply_shard_filter_zero_or_negative_is_passthrough():
    """Defensive: bogus N (caught by CLI validation) does not crash here."""
    keys = _mk_keys(5)
    assert pl._apply_shard_filter(keys, 0, 0) is keys
    assert pl._apply_shard_filter(keys, 0, -1) is keys


def test_process_provider_emits_shard_manifest(monkeypatch, caplog):
    """`shard i/N: K files, first=… last=…` INFO line at startup when N>1."""
    keys = _mk_keys(8)
    monkeypatch.setattr(s3io, "list_input_keys", lambda **_: iter(keys))
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    cfg = pl.PipelineConfig(
        input_bucket="i",
        output_bucket="o",
        input_prefix="",
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_revision=None,
        level="text",
        chunking_strategy_name="semantic",
        force=True,  # bypass skip-check entirely; we don't actually run the encode loop
        encoder=em.EncoderConfig(
            batch_size=2, min_char_length=5, content_types=frozenset({"ar"})
        ),
        shard_index=2,
        num_shards=4,
    )
    # Stub the real pipeline body so we never actually download/encode/upload.
    # `process_provider` returns early after the manifest line if to_process is
    # falsy — but with force=True every key lands in to_process. Patch the
    # encode/upload helpers to no-ops so the loop is a no-op too.
    monkeypatch.setattr(pl, "_encode_to_local", lambda *a, **k: (0, {}))
    monkeypatch.setattr(s3io, "download_to_local", lambda *a, **k: None)
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)

    with caplog.at_level("INFO", logger="impresso_text_embedder.pipeline"):
        pl.process_provider("SNL", MagicMock(), cfg)

    [manifest] = [r.message for r in caplog.records if r.message.startswith("shard ")]
    # Round-robin: shard 2 of 4 from 8 keys → indices 2 and 6 (years 1902, 1906).
    assert "shard 2/4: 2 files" in manifest
    assert "first=SNL/EXP/EXP-1902.jsonl.bz2" in manifest
    assert "last=SNL/EXP/EXP-1906.jsonl.bz2" in manifest


def test_process_provider_no_manifest_when_unsharded(monkeypatch, caplog):
    """Default num_shards=1 → no `shard …` log line (unchanged behaviour)."""
    monkeypatch.setattr(s3io, "list_input_keys", lambda **_: iter(_mk_keys(3)))
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    monkeypatch.setattr(pl, "_encode_to_local", lambda *a, **k: (0, {}))
    monkeypatch.setattr(s3io, "download_to_local", lambda *a, **k: None)
    monkeypatch.setattr(s3io, "upload_local_file", lambda *a, **k: None)
    with caplog.at_level("INFO", logger="impresso_text_embedder.pipeline"):
        pl.process_provider("SNL", MagicMock(), _cfg(force=True))
    assert not any(r.message.startswith("shard ") for r in caplog.records)


def test_process_provider_empty_shard_logs_no_work(monkeypatch, caplog):
    """An empty shard (filter yields zero keys) logs `0 files (no work)`."""
    # 3 keys, sharded into 4 partitions → shard 3 ends up empty.
    keys = _mk_keys(3)
    monkeypatch.setattr(s3io, "list_input_keys", lambda **_: iter(keys))
    monkeypatch.setattr(s3io, "head_last_modified", lambda b, k: None)
    cfg = pl.PipelineConfig(
        input_bucket="i",
        output_bucket="o",
        input_prefix="",
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_revision=None,
        level="text",
        chunking_strategy_name="semantic",
        force=True,
        encoder=em.EncoderConfig(
            batch_size=2, min_char_length=5, content_types=frozenset({"ar"})
        ),
        shard_index=3,
        num_shards=4,
    )
    with caplog.at_level("INFO", logger="impresso_text_embedder.pipeline"):
        summary = pl.process_provider("SNL", MagicMock(), cfg)
    assert summary["processed"] == 0
    [manifest] = [r.message for r in caplog.records if r.message.startswith("shard ")]
    assert manifest == "shard 3/4: 0 files (no work)"
