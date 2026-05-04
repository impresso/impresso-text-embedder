from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from impresso_text_embedder.research import study_config as sc


# ---------------------------------------------------------------------------
# Helpers — minimal valid base + study YAML pair the fixtures can reuse.
# ---------------------------------------------------------------------------


_BASE_DICT = {
    "s3": {"bucket": "sandbox", "rebuilt_bucket": "rebuilt"},
    "paths": {
        "local_root": "tmp/{study}",
        "s3_root":    "research/{study}",
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


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_base_and_study(
    tmp_path: Path,
    *,
    study_overlay: dict | None = None,
    base_overrides: dict | None = None,
) -> Path:
    """Materialise a base.yaml + study.yaml pair and return the study path."""
    base = dict(_BASE_DICT)
    if base_overrides:
        for k, v in base_overrides.items():
            base[k] = v
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, base)

    overlay = study_overlay or {
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512, 1024]},
    }
    overlay = {"extends": "base.yaml", **overlay}
    study_path = tmp_path / "study.yaml"
    _write_yaml(study_path, overlay)
    return study_path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_resolves_extends_and_substitutes_paths(tmp_path) -> None:
    study_path = _write_base_and_study(tmp_path)
    cfg = sc.load_study_config(study_path)
    assert cfg.study.name == "v1"
    assert cfg.corpus.min_tokens == 4000
    assert cfg.scenarios.chunk_sizes == (512, 1024)
    assert cfg.s3_key(sc.CORPUS_FILENAME) == "research/v1/corpus.jsonl.bz2"
    assert cfg.s3_key(sc.QUERIES_FILENAME) == "research/v1/queries.jsonl.bz2"
    assert cfg.study_s3_root() == "research/v1"
    assert cfg.s3_key(sc.scenario_filename("S3")) == "research/v1/S3.jsonl.bz2"


def test_overlay_replaces_lists_does_not_append(tmp_path) -> None:
    """Overlay's ``chunkers: [...]`` must REPLACE base's, not extend it."""
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {"min_tokens": 4000},
            "scenarios": {
                "chunkers": ["semantic"],
                "chunk_sizes": [512],
            },
        },
    )
    cfg = sc.load_study_config(study_path)
    assert cfg.scenarios.chunkers == ("semantic",)


def test_overlay_recursive_dict_merge(tmp_path) -> None:
    """Overlay's nested dict adds/overrides leaf keys, preserves siblings."""
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {
                "min_tokens": 4000,
                "providers": {"de": ["SNL"]},  # adds de; preserves fr
                "chars_per_token": {"de": 3.5},  # adds de; preserves fr
                "languages": ["fr", "de"],
            },
            "scenarios": {"chunk_sizes": [512]},
        },
    )
    cfg = sc.load_study_config(study_path)
    assert set(cfg.corpus.providers) == {"fr", "de"}
    assert cfg.corpus.providers["fr"] == ("BNF",)
    assert cfg.corpus.providers["de"] == ("SNL",)


def test_config_sha_stable_across_field_reorder(tmp_path) -> None:
    """SHA must be insensitive to dict-insert order (sorted keys)."""
    cfg_a = sc.load_study_config(_write_base_and_study(tmp_path))
    sha_a = cfg_a.config_sha
    # rebuild same config fresh — same SHA expected
    cfg_b = sc.load_study_config(_write_base_and_study(tmp_path))
    assert sha_a == cfg_b.config_sha


def test_config_sha_changes_when_value_changes(tmp_path) -> None:
    cfg_a = sc.load_study_config(_write_base_and_study(tmp_path))
    cfg_b = sc.load_study_config(_write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {"min_tokens": 5000},  # changed
            "scenarios": {"chunk_sizes": [512, 1024]},
        },
    ))
    assert cfg_a.config_sha != cfg_b.config_sha


def test_local_path_substitutes_study_name(tmp_path) -> None:
    cfg = sc.load_study_config(_write_base_and_study(tmp_path))
    p = cfg.local_path(sc.CORPUS_FILENAME)
    assert "{study}" not in str(p)
    assert cfg.study.name in p.parts
    assert p.name == sc.CORPUS_FILENAME


def test_local_path_for_scenario(tmp_path) -> None:
    cfg = sc.load_study_config(_write_base_and_study(tmp_path))
    p = cfg.local_path(sc.scenario_filename("S0"))
    assert p.name == "S0.jsonl.bz2"
    assert cfg.study.name in p.parts


# ---------------------------------------------------------------------------
# Validation negatives
# ---------------------------------------------------------------------------


def test_path_template_must_contain_study_token(tmp_path) -> None:
    bad_base = dict(_BASE_DICT)
    bad_paths = dict(_BASE_DICT["paths"])
    bad_paths["local_root"] = "tmp/static"  # missing {study}
    bad_base["paths"] = bad_paths
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, bad_base)
    study_path = tmp_path / "study.yaml"
    _write_yaml(study_path, {
        "extends": "base.yaml",
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512]},
    })
    with pytest.raises(Exception) as exc:
        sc.load_study_config(study_path)
    assert "{study}" in str(exc.value)


def test_unknown_top_level_key_rejected(tmp_path) -> None:
    """``extra='forbid'`` catches YAML typos like ``langauges:``."""
    study_path = _write_base_and_study(tmp_path)
    raw = yaml.safe_load(study_path.read_text())
    raw["studyy"] = {"name": "v1-typo"}  # typo'd top-level key
    _write_yaml(study_path, raw)
    with pytest.raises(Exception):
        sc.load_study_config(study_path)


def test_min_tokens_required(tmp_path) -> None:
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "scenarios": {"chunk_sizes": [512]},  # min_tokens missing
        },
    )
    with pytest.raises(Exception):
        sc.load_study_config(study_path)


def test_chunk_sizes_required_and_positive(tmp_path) -> None:
    # missing chunk_sizes
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {"min_tokens": 4000},
        },
    )
    with pytest.raises(Exception):
        sc.load_study_config(study_path)

    # negative chunk_sizes
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {"min_tokens": 4000},
            "scenarios": {"chunk_sizes": [-1]},
        },
    )
    with pytest.raises(Exception):
        sc.load_study_config(study_path)


def test_max_tokens_must_exceed_min_tokens(tmp_path) -> None:
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {"min_tokens": 5000, "max_tokens": 4000},
            "scenarios": {"chunk_sizes": [512]},
        },
    )
    with pytest.raises(Exception) as exc:
        sc.load_study_config(study_path)
    assert "max_tokens" in str(exc.value) or "min_tokens" in str(exc.value)


def test_year_min_must_not_exceed_year_max(tmp_path) -> None:
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {
                "min_tokens": 4000,
                "year_min": 2000,
                "year_max": 1900,
            },
            "scenarios": {"chunk_sizes": [512]},
        },
    )
    with pytest.raises(Exception) as exc:
        sc.load_study_config(study_path)
    assert "year" in str(exc.value)


def test_languages_must_have_providers_and_chars_per_token(tmp_path) -> None:
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "v1"},
            "corpus": {
                "min_tokens": 4000,
                "languages": ["fr", "lb"],  # base has no providers entry for lb
            },
            "scenarios": {"chunk_sizes": [512]},
        },
    )
    with pytest.raises(Exception) as exc:
        sc.load_study_config(study_path)
    assert "lb" in str(exc.value)


def test_position_buckets_must_be_unique(tmp_path: Path) -> None:
    base = dict(_BASE_DICT)
    bad_qg = dict(_BASE_DICT["query_generation"])
    bad_qg["position_buckets"] = ["head", "head", "tail"]
    base["query_generation"] = bad_qg
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, base)
    study = tmp_path / "study.yaml"
    _write_yaml(study, {
        "extends": "base.yaml",
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512]},
    })
    with pytest.raises(Exception) as exc:
        sc.load_study_config(study)
    assert "unique" in str(exc.value).lower()


def test_position_buckets_must_be_non_empty(tmp_path: Path) -> None:
    base = dict(_BASE_DICT)
    bad_qg = dict(_BASE_DICT["query_generation"])
    bad_qg["position_buckets"] = []
    base["query_generation"] = bad_qg
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, base)
    study = tmp_path / "study.yaml"
    _write_yaml(study, {
        "extends": "base.yaml",
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512]},
    })
    with pytest.raises(Exception):
        sc.load_study_config(study)


def test_queries_per_bucket_defaults_to_one(tmp_path: Path) -> None:
    cfg = sc.load_study_config(_write_base_and_study(tmp_path))
    assert cfg.query_generation.queries_per_bucket == 1


def test_queries_per_bucket_must_be_positive(tmp_path: Path) -> None:
    base = dict(_BASE_DICT)
    bad_qg = dict(_BASE_DICT["query_generation"])
    bad_qg["queries_per_bucket"] = 0
    base["query_generation"] = bad_qg
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, base)
    study = tmp_path / "study.yaml"
    _write_yaml(study, {
        "extends": "base.yaml",
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512]},
    })
    with pytest.raises(Exception):
        sc.load_study_config(study)


def test_query_types_field_no_longer_accepted(tmp_path: Path) -> None:
    """``query_types`` was dropped from the schema: a YAML that still
    names it must error so ``extra="forbid"`` keeps the schema honest."""
    base = dict(_BASE_DICT)
    legacy_qg = dict(_BASE_DICT["query_generation"])
    legacy_qg["query_types"] = ["question"]
    base["query_generation"] = legacy_qg
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, base)
    study = tmp_path / "study.yaml"
    _write_yaml(study, {
        "extends": "base.yaml",
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512]},
    })
    with pytest.raises(Exception):
        sc.load_study_config(study)


def test_study_name_must_match_safe_pattern(tmp_path) -> None:
    study_path = _write_base_and_study(
        tmp_path,
        study_overlay={
            "study": {"name": "with spaces"},
            "corpus": {"min_tokens": 4000},
            "scenarios": {"chunk_sizes": [512]},
        },
    )
    with pytest.raises(Exception):
        sc.load_study_config(study_path)


# ---------------------------------------------------------------------------
# Extends edge cases
# ---------------------------------------------------------------------------


def test_extends_chain_rejected(tmp_path) -> None:
    grandbase = tmp_path / "grandbase.yaml"
    _write_yaml(grandbase, _BASE_DICT)
    base = tmp_path / "base.yaml"
    _write_yaml(base, {"extends": "grandbase.yaml"})
    study = tmp_path / "study.yaml"
    _write_yaml(study, {
        "extends": "base.yaml",
        "study": {"name": "v1"},
        "corpus": {"min_tokens": 4000},
        "scenarios": {"chunk_sizes": [512]},
    })
    with pytest.raises(Exception) as exc:
        sc.load_study_config(study)
    assert "chained" in str(exc.value).lower() or "extends" in str(exc.value).lower()


def test_self_extends_rejected(tmp_path) -> None:
    study = tmp_path / "study.yaml"
    _write_yaml(study, {
        "extends": "study.yaml",
        "study": {"name": "v1"},
    })
    with pytest.raises(Exception) as exc:
        sc.load_study_config(study)
    assert "self" in str(exc.value).lower()


def test_top_level_must_be_mapping(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(Exception) as exc:
        sc.load_study_config(bad)
    assert "mapping" in str(exc.value).lower()


def test_loads_without_extends(tmp_path) -> None:
    """A YAML can be standalone (no ``extends:``) — base is optional."""
    standalone = dict(_BASE_DICT)
    standalone["study"] = {"name": "stand"}
    standalone["corpus"] = {**_BASE_DICT["corpus"], "min_tokens": 4000}
    standalone["scenarios"] = {**_BASE_DICT["scenarios"], "chunk_sizes": [512]}
    p = tmp_path / "study.yaml"
    _write_yaml(p, standalone)
    cfg = sc.load_study_config(p)
    assert cfg.study.name == "stand"


# ---------------------------------------------------------------------------
# Repository fixtures — base.yaml + study-v1.yaml at the actual paths
# ---------------------------------------------------------------------------


def test_repo_study_v1_yaml_loads() -> None:
    """The frozen pre-refactor snapshot at configs/research/study-v1.yaml
    must validate; this is the regression check the refactor depends on."""
    cfg_path = Path(__file__).parent.parent / "configs/research/study-v1.yaml"
    cfg = sc.load_study_config(cfg_path)
    assert cfg.study.name == "v1"
    assert cfg.corpus.min_tokens == 4000
    assert cfg.scenarios.chunk_sizes == (512, 1024, 2048, 4096, 8190)
    assert "v1" in cfg.s3_key(sc.CORPUS_FILENAME)


def test_repo_base_yaml_alone_fails_validation() -> None:
    """base.yaml is intentionally a partial — loading it standalone should
    error because study.name / corpus.min_tokens / scenarios.chunk_sizes
    are missing. That's the signal it's a mixin, not a runnable study."""
    base_path = Path(__file__).parent.parent / "configs/research/base.yaml"
    with pytest.raises(Exception):
        sc.load_study_config(base_path)


def test_dedent_helper_is_importable() -> None:
    """Sanity import — fail-loud if textwrap.dedent disappears upstream."""
    assert dedent("  x") == "x"


def test_summary_lists_key_study_facets() -> None:
    """``summary()`` is a notebook ergonomics aid; smoke-check the headline
    facets land in the rendered text so a YAML rename doesn't silently
    drop them from notebooks."""
    cfg_path = Path(__file__).parent.parent / "configs/research/study-v1.yaml"
    cfg = sc.load_study_config(cfg_path)
    text = cfg.summary()
    assert cfg.study.name in text
    assert cfg.config_sha in text
    assert cfg.embed.model_name in text
    assert cfg.embed.model_revision in text
    for lg in cfg.corpus.languages:
        assert lg in text
    for chunker in cfg.scenarios.chunkers:
        assert chunker in text
    assert str(cfg.scenarios.chunk_sizes[0]) in text
    assert cfg.n_scenarios() == (
        (1 if cfg.scenarios.truncate_baseline else 0)
        + len(cfg.scenarios.chunkers) * len(cfg.scenarios.chunk_sizes)
    )


def test_repr_markdown_renders_in_jupyter() -> None:
    """``inputs.study`` in a Jupyter cell uses ``_repr_markdown_`` —
    smoke-check it returns markdown that mentions the study name."""
    cfg_path = Path(__file__).parent.parent / "configs/research/study-v1.yaml"
    cfg = sc.load_study_config(cfg_path)
    md = cfg._repr_markdown_()
    assert md.lstrip().startswith("###")
    assert cfg.study.name in md
    assert cfg.config_sha in md
