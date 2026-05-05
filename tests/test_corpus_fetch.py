from __future__ import annotations

import bz2
import json
from pathlib import Path

import orjson
import pytest

from impresso_text_embedder.research import corpus_fetch


def _write_manifest(path: Path, entries: list[dict]) -> None:
    with path.open("wb") as fh:
        for e in entries:
            fh.write(orjson.dumps(e, option=orjson.OPT_APPEND_NEWLINE))


def _entry(
    *,
    ci_id: str,
    lg: str = "fr",
    year: int = 1920,
    len_chars: int = 20000,
    ocrqa: float = 0.95,
    provider: str = "BNF",
    alias: str = "jdpl",
) -> dict:
    return {
        "ci_id": ci_id,
        "lg": lg,
        "year": year,
        "len_chars": len_chars,
        "ocrqa": ocrqa,
        "provider": provider,
        "alias": alias,
        "rebuilt_bucket": "122-rebuilt-final",
        "rebuilt_key": f"{provider}/{alias}/{alias}-{year}.jsonl.bz2",
    }


def _fake_iter(rebuilt_by_key: dict[tuple[str, str], list[dict]]):
    """Build a stub for ``s3io.iter_jsonl_bz2`` over an in-memory mapping.

    Yields one decoded line per record (no bz2 wrapping needed since the
    real implementation already returns decoded strings).
    """

    def iter_jsonl_bz2(bucket: str, key: str):
        for rec in rebuilt_by_key.get((bucket, key), []):
            yield json.dumps(rec)

    return iter_jsonl_bz2


def _read_output_records(path: Path) -> list[dict]:
    with bz2.open(path, "rb") as fh:
        return [orjson.loads(line) for line in fh if line.strip()]


class TestReadManifest:
    def test_round_trip(self, tmp_path: Path):
        manifest = [_entry(ci_id="a"), _entry(ci_id="b", lg="de", year=1950)]
        path = tmp_path / "manifest.jsonl"
        _write_manifest(path, manifest)
        loaded = corpus_fetch.read_manifest(path)
        assert [e.ci_id for e in loaded] == ["a", "b"]
        assert loaded[1].lg == "de"
        assert loaded[1].year == 1950

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "manifest.jsonl"
        with path.open("wb") as fh:
            fh.write(orjson.dumps(_entry(ci_id="a"), option=orjson.OPT_APPEND_NEWLINE))
            fh.write(b"\n")
            fh.write(orjson.dumps(_entry(ci_id="b"), option=orjson.OPT_APPEND_NEWLINE))
        loaded = corpus_fetch.read_manifest(path)
        assert [e.ci_id for e in loaded] == ["a", "b"]


class TestGroupBySource:
    def test_groups_by_bucket_key(self):
        a = corpus_fetch.ManifestEntry.from_dict(
            _entry(ci_id="a", provider="BNF", alias="jdpl", year=1900)
        )
        b = corpus_fetch.ManifestEntry.from_dict(
            _entry(ci_id="b", provider="BNF", alias="jdpl", year=1900)
        )
        c = corpus_fetch.ManifestEntry.from_dict(
            _entry(ci_id="c", provider="SNL", alias="DTT", year=1950)
        )
        groups = corpus_fetch._group_by_source([a, b, c])
        assert set(groups.keys()) == {
            ("122-rebuilt-final", "BNF/jdpl/jdpl-1900.jsonl.bz2"),
            ("122-rebuilt-final", "SNL/DTT/DTT-1950.jsonl.bz2"),
        }
        first_group = groups[("122-rebuilt-final", "BNF/jdpl/jdpl-1900.jsonl.bz2")]
        assert set(first_group.keys()) == {"a", "b"}


class TestCorpusRecordFromManifest:
    def test_uses_record_ft_when_present(self):
        entry = corpus_fetch.ManifestEntry.from_dict(_entry(ci_id="a"))
        rec = {"id": "a", "tp": "ar", "ft": "Hello world.", "sents": [{"tok": []}]}
        out, reconstructed = corpus_fetch.CorpusRecord.from_manifest(entry, rec)
        assert out.ft == "Hello world."
        assert out.sents == [{"tok": []}]
        assert out.tp == "ar"
        assert out.lg == "fr"
        assert out.provider == "BNF"
        assert reconstructed is False

    def test_reconstructs_ft_from_offsets(self):
        entry = corpus_fetch.ManifestEntry.from_dict(_entry(ci_id="a"))
        rec = {
            "id": "a",
            "tp": "ar",
            "sents": [
                {"tok": [{"t": "Hello", "o": 0}, {"t": "world", "o": 6}]},
            ],
        }
        out, reconstructed = corpus_fetch.CorpusRecord.from_manifest(entry, rec)
        assert out.ft == "Hello world"
        assert reconstructed is True

    def test_passes_lingproc_path_when_present(self):
        entry = corpus_fetch.ManifestEntry.from_dict(_entry(ci_id="a"))
        rec = {"id": "a", "tp": "ar", "ft": "x", "lingproc_path": "s3://x/y"}
        out, _ = corpus_fetch.CorpusRecord.from_manifest(entry, rec)
        assert out.lingproc_path == "s3://x/y"
        assert out.to_jsonable()["lingproc_path"] == "s3://x/y"

    def test_omits_lingproc_path_when_absent(self):
        entry = corpus_fetch.ManifestEntry.from_dict(_entry(ci_id="a"))
        rec = {"id": "a", "tp": "ar", "ft": "x"}
        out, _ = corpus_fetch.CorpusRecord.from_manifest(entry, rec)
        assert out.lingproc_path is None
        assert "lingproc_path" not in out.to_jsonable()


class TestFetchCorpus:
    def test_writes_records_in_manifest_order(self, tmp_path: Path, monkeypatch):
        # Three manifest entries spread across two rebuilt files. The output
        # must preserve the manifest's order even though the workers complete
        # out of order.
        manifest_entries = [
            _entry(ci_id="a-1880", year=1880),
            _entry(ci_id="b-1950", lg="de", year=1950, provider="SNL", alias="DTT", len_chars=15000),
            _entry(ci_id="a-1880-second", year=1880),
        ]
        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(manifest_path, manifest_entries)

        rebuilt = {
            ("122-rebuilt-final", "BNF/jdpl/jdpl-1880.jsonl.bz2"): [
                {"id": "noise", "tp": "ar", "ft": "skip me"},
                {"id": "a-1880", "tp": "ar", "ft": "alpha"},
                {"id": "a-1880-second", "tp": "ar", "ft": "alpha-second"},
            ],
            ("122-rebuilt-final", "SNL/DTT/DTT-1950.jsonl.bz2"): [
                {"id": "b-1950", "tp": "ar", "ft": "beta"},
            ],
        }
        monkeypatch.setattr(
            corpus_fetch.s3io, "iter_jsonl_bz2", _fake_iter(rebuilt)
        )

        local_out = tmp_path / "out.jsonl.bz2"
        stats = corpus_fetch.fetch_corpus(manifest_path, local_out, max_workers=2)

        assert stats.manifest_entries == 3
        assert stats.rebuilt_files == 2
        assert stats.written == 3
        assert stats.ft_from_record == 3
        assert stats.ft_reconstructed == 0
        assert stats.missing_in_rebuilt == []

        records = _read_output_records(local_out)
        assert [r["ci_id"] for r in records] == ["a-1880", "b-1950", "a-1880-second"]
        assert records[0]["ft"] == "alpha"
        assert records[1]["ft"] == "beta"
        assert records[1]["lg"] == "de"

    def test_records_missing_ci_ids(self, tmp_path: Path, monkeypatch):
        manifest_entries = [
            _entry(ci_id="present"),
            _entry(ci_id="ghost"),
        ]
        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(manifest_path, manifest_entries)

        rebuilt = {
            ("122-rebuilt-final", "BNF/jdpl/jdpl-1920.jsonl.bz2"): [
                {"id": "present", "tp": "ar", "ft": "ok"},
            ],
        }
        monkeypatch.setattr(
            corpus_fetch.s3io, "iter_jsonl_bz2", _fake_iter(rebuilt)
        )

        stats = corpus_fetch.fetch_corpus(
            manifest_path, tmp_path / "out.jsonl.bz2", max_workers=1
        )
        assert stats.written == 1
        assert stats.missing_in_rebuilt == ["ghost"]

    def test_reconstructs_ft_when_record_lacks_it(self, tmp_path: Path, monkeypatch):
        manifest_entries = [_entry(ci_id="a")]
        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(manifest_path, manifest_entries)

        rebuilt = {
            ("122-rebuilt-final", "BNF/jdpl/jdpl-1920.jsonl.bz2"): [
                {
                    "id": "a",
                    "tp": "ar",
                    "sents": [
                        {"tok": [{"t": "Hello", "o": 0}, {"t": "world", "o": 6}]},
                    ],
                },
            ],
        }
        monkeypatch.setattr(
            corpus_fetch.s3io, "iter_jsonl_bz2", _fake_iter(rebuilt)
        )

        stats = corpus_fetch.fetch_corpus(
            manifest_path, tmp_path / "out.jsonl.bz2", max_workers=1
        )
        assert stats.ft_reconstructed == 1
        assert stats.ft_from_record == 0
        records = _read_output_records(tmp_path / "out.jsonl.bz2")
        assert records[0]["ft"] == "Hello world"

    def test_empty_manifest_writes_empty_shard(self, tmp_path: Path, monkeypatch):
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_bytes(b"")
        # The fake iter should never be called; we still patch to be safe.
        monkeypatch.setattr(
            corpus_fetch.s3io, "iter_jsonl_bz2", _fake_iter({})
        )
        out = tmp_path / "out.jsonl.bz2"
        stats = corpus_fetch.fetch_corpus(manifest_path, out, max_workers=1)
        assert stats.manifest_entries == 0
        assert stats.written == 0
        assert out.exists()


def _write_study_config(tmp_path: Path, *, local_root: Path, s3_root: str = "test/{study}") -> Path:
    """Materialise a minimal study YAML pointing local_root at ``tmp_path``.

    The manifest path the CLI consumes will be ``<local_root>/manifest.jsonl``;
    callers write their manifest there before invoking ``corpus_fetch.main``.
    """
    import yaml

    base = {
        "s3": {"bucket": "test-bucket", "rebuilt_bucket": "122-rebuilt-final"},
        "paths": {
            "local_root": str(local_root) + "/{study}",
            "s3_root": s3_root,
        },
        "corpus": {
            "input_path": "tmp/agg.jsonl",
            "languages": ["fr"],
            "ocrqa_min": 0.9,
            "year_min": 1880,
            "year_max": 1980,
            "providers": {"fr": ["BNF"]},
            "chars_per_token": {"fr": 4.5},
            "n_per_lg": 100,
        },
        "embed": {
            "model_name": "Alibaba-NLP/gte-multilingual-base",
            "model_revision": "abc",
        },
        "scenarios": {"chunkers": ["fixed-window"], "aggregator": "mean"},
        "query_generation": {
            "endpoint": "https://example.test/v1",
            "model": "test-model",
            "max_parallel": 2,
            "temperature": 0.7,
            "max_output_tokens": 1500,
            "request_timeout_s": 60.0,
            "retry_attempts": 3,
            "position_buckets": ["head", "mid", "tail"],
            "queries_per_bucket": 1,
        },
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    study = {
        "extends": "base.yaml",
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512]},
    }
    study_path = tmp_path / "study.yaml"
    study_path.write_text(yaml.safe_dump(study), encoding="utf-8")
    return study_path


class TestMainCli:
    def test_no_upload_writes_to_local_mirror(self, tmp_path: Path, monkeypatch):
        config_path = _write_study_config(tmp_path, local_root=tmp_path / "out")
        manifest_path = tmp_path / "out" / "v1" / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_manifest(manifest_path, [_entry(ci_id="a")])
        rebuilt = {
            ("122-rebuilt-final", "BNF/jdpl/jdpl-1920.jsonl.bz2"): [
                {"id": "a", "tp": "ar", "ft": "ok"},
            ],
        }
        monkeypatch.setattr(
            corpus_fetch.s3io, "iter_jsonl_bz2", _fake_iter(rebuilt)
        )
        upload_called = []
        monkeypatch.setattr(
            corpus_fetch.s3io,
            "upload_local_file",
            lambda *a, **kw: upload_called.append((a, kw)),
        )

        rc = corpus_fetch.main(["--config", str(config_path), "--no-upload"])
        assert rc == 0
        assert upload_called == []
        # Mirror lands at <local_root>/<study>/corpus.jsonl.bz2
        mirror = tmp_path / "out" / "v1" / "corpus.jsonl.bz2"
        assert mirror.exists()

    def test_upload_path_uses_study_s3_key(self, tmp_path: Path, monkeypatch):
        config_path = _write_study_config(
            tmp_path, local_root=tmp_path / "out", s3_root="research/{study}"
        )
        manifest_path = tmp_path / "out" / "v1" / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_manifest(manifest_path, [_entry(ci_id="a")])
        rebuilt = {
            ("122-rebuilt-final", "BNF/jdpl/jdpl-1920.jsonl.bz2"): [
                {"id": "a", "tp": "ar", "ft": "ok"},
            ],
        }
        monkeypatch.setattr(
            corpus_fetch.s3io, "iter_jsonl_bz2", _fake_iter(rebuilt)
        )

        captured: dict = {}

        def fake_upload(local_path, bucket, key):
            captured["local_path"] = Path(local_path)
            captured["bucket"] = bucket
            captured["key"] = key
            assert captured["local_path"].exists(), "upload before file written"

        monkeypatch.setattr(corpus_fetch.s3io, "upload_local_file", fake_upload)

        rc = corpus_fetch.main(["--config", str(config_path)])
        assert rc == 0
        assert captured["bucket"] == "test-bucket"
        # Output key is derived from s3_root + corpus.jsonl.bz2 — no per-CLI flag.
        assert captured["key"] == "research/v1/corpus.jsonl.bz2"
