"""Unit tests for the ``impresso-research-study-seed`` CLI.

The CLI wraps :func:`io.copy_s3_object`; tests stub that helper and
assert the right (src, dst) pairs are computed from the two study
configs and that the no-op safeguards (identical study names, cross
buckets, --dry-run) fire before any S3 call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from impresso_text_embedder.research import study_seed


# Reuse the same minimal-base shape as test_study_config so the two
# test modules share assumptions about what a valid study YAML looks
# like.
_BASE_DICT = {
    "s3": {"bucket": "sandbox", "rebuilt_bucket": "rebuilt"},
    "paths": {
        "local_root": "tmp/{study}",
        "s3_root": "research/{study}",
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
        "model_revision": "abc123",
    },
    "scenarios": {
        "chunkers": ["fixed-window"],
        "aggregator": "mean",
    },
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


def _write_pair(tmp_path: Path, src_name: str, dst_name: str) -> tuple[Path, Path]:
    """Write base.yaml + two study YAMLs sharing it, return (src_path, dst_path)."""
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_BASE_DICT), encoding="utf-8")

    def _write_study(name: str) -> Path:
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump({
                "extends": "base.yaml",
                "study": {"name": name},
                "corpus": {"min_tokens": 4000},
                "scenarios": {"chunk_sizes": [512]},
            }),
            encoding="utf-8",
        )
        return path

    return _write_study(src_name), _write_study(dst_name)


def test_dry_run_prints_pairs_and_calls_no_copy(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    src_path, dst_path = _write_pair(tmp_path, "src-study", "dst-study")
    calls: list[tuple] = []

    def fake_copy(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(study_seed, "copy_s3_object", fake_copy)

    rc = study_seed.main([
        "--source-config", str(src_path),
        "--target-config", str(dst_path),
        "--dry-run",
    ])
    assert rc == 0
    assert calls == [], "--dry-run must not call copy_s3_object"

    out = capsys.readouterr().out
    # Three default artefacts: corpus, queries, queries-embedded.
    assert "research/src-study/corpus.jsonl.bz2" in out
    assert "research/dst-study/corpus.jsonl.bz2" in out
    assert "research/src-study/queries.jsonl.bz2" in out
    assert "research/dst-study/queries.jsonl.bz2" in out
    assert "research/src-study/queries-embedded.jsonl.bz2" in out
    assert "research/dst-study/queries-embedded.jsonl.bz2" in out


def test_run_invokes_copy_for_each_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    src_path, dst_path = _write_pair(tmp_path, "src", "dst")
    calls: list[dict] = []

    def fake_copy(src_bucket, src_key, dst_bucket, dst_key, *, overwrite=False):
        calls.append({
            "src_bucket": src_bucket,
            "src_key": src_key,
            "dst_bucket": dst_bucket,
            "dst_key": dst_key,
            "overwrite": overwrite,
        })
        return True

    monkeypatch.setattr(study_seed, "copy_s3_object", fake_copy)

    rc = study_seed.main([
        "--source-config", str(src_path),
        "--target-config", str(dst_path),
    ])
    assert rc == 0
    assert len(calls) == 3
    for c in calls:
        assert c["src_bucket"] == "sandbox"
        assert c["dst_bucket"] == "sandbox"
        assert c["src_key"].startswith("research/src/")
        assert c["dst_key"].startswith("research/dst/")
        assert c["overwrite"] is False


def test_overwrite_flag_propagates(tmp_path: Path, monkeypatch) -> None:
    src_path, dst_path = _write_pair(tmp_path, "src", "dst")
    seen: list[bool] = []

    def fake_copy(src_bucket, src_key, dst_bucket, dst_key, *, overwrite=False):
        seen.append(overwrite)
        return True

    monkeypatch.setattr(study_seed, "copy_s3_object", fake_copy)

    study_seed.main([
        "--source-config", str(src_path),
        "--target-config", str(dst_path),
        "--overwrite",
    ])
    assert seen == [True, True, True]


def test_artifacts_subset_filter(tmp_path: Path, monkeypatch) -> None:
    src_path, dst_path = _write_pair(tmp_path, "src", "dst")
    keys: list[str] = []

    def fake_copy(src_bucket, src_key, dst_bucket, dst_key, *, overwrite=False):
        keys.append(dst_key)
        return True

    monkeypatch.setattr(study_seed, "copy_s3_object", fake_copy)

    study_seed.main([
        "--source-config", str(src_path),
        "--target-config", str(dst_path),
        "--artifacts", "queries,queries-embedded",
    ])
    assert len(keys) == 2
    assert all("corpus" not in k for k in keys)


def test_unknown_artifact_name_rejected(tmp_path: Path) -> None:
    src_path, dst_path = _write_pair(tmp_path, "src", "dst")
    with pytest.raises(SystemExit) as exc:
        study_seed.main([
            "--source-config", str(src_path),
            "--target-config", str(dst_path),
            "--artifacts", "corpus,not-an-artifact",
        ])
    assert "not-an-artifact" in str(exc.value)


def test_identical_study_names_rejected(tmp_path: Path) -> None:
    src_path, dst_path = _write_pair(tmp_path, "same", "same")
    with pytest.raises(SystemExit) as exc:
        study_seed.main([
            "--source-config", str(src_path),
            "--target-config", str(dst_path),
        ])
    assert "identical" in str(exc.value).lower()


def test_cross_bucket_rejected(tmp_path: Path) -> None:
    """Two YAMLs with different s3.bucket values must error early."""
    src_path, dst_path = _write_pair(tmp_path, "src", "dst")
    # Rewrite dst with a different bucket; it can't extend base anymore
    # because base has the wrong bucket — write standalone instead.
    standalone = dict(_BASE_DICT)
    standalone["s3"] = {"bucket": "other-sandbox", "rebuilt_bucket": "rebuilt"}
    standalone["study"] = {"name": "dst"}
    standalone["corpus"] = {**_BASE_DICT["corpus"], "min_tokens": 4000}
    standalone["scenarios"] = {**_BASE_DICT["scenarios"], "chunk_sizes": [512]}
    dst_path.write_text(yaml.safe_dump(standalone), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        study_seed.main([
            "--source-config", str(src_path),
            "--target-config", str(dst_path),
        ])
    assert "cross-bucket" in str(exc.value)


def test_resolve_pairs_returns_correct_filenames(tmp_path: Path) -> None:
    """Direct unit test of the path-mapping helper, no CLI involved."""
    from impresso_text_embedder.research.study_config import load_study_config

    src_path, dst_path = _write_pair(tmp_path, "src", "dst")
    src_cfg = load_study_config(src_path)
    dst_cfg = load_study_config(dst_path)
    pairs = study_seed._resolve_pairs(
        src_cfg, dst_cfg, ("corpus", "queries", "queries-embedded")
    )
    assert pairs == [
        ("sandbox", "research/src/corpus.jsonl.bz2",
         "sandbox", "research/dst/corpus.jsonl.bz2"),
        ("sandbox", "research/src/queries.jsonl.bz2",
         "sandbox", "research/dst/queries.jsonl.bz2"),
        ("sandbox", "research/src/queries-embedded.jsonl.bz2",
         "sandbox", "research/dst/queries-embedded.jsonl.bz2"),
    ]
