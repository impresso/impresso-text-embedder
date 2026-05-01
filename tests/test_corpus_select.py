from __future__ import annotations

import json
from pathlib import Path

import pytest

from impresso_text_embedder.research import corpus_select


def _agg_record(
    *,
    ci_id: str,
    lg: str = "fr",
    year: int = 1920,
    tp: str = "article",
    len_chars: int = 20000,
    ocrqa: float = 0.95,
    provider: str = "LeTemps",
    alias: str = "LET",
) -> dict:
    """Build a single aggregator record matching the schema observed in the wild."""
    source_key = f"langident/langident-lid-ensemble_multilingual_v2-0-2/{provider}/{alias}/{alias}-{year}.jsonl.bz2"
    return {
        "id": ci_id,
        "year": str(year),
        "lg": lg,
        "len": len_chars,
        "lg_decision": "all",
        "tp": tp,
        "alphabetical_ratio": 0.85,
        "ocrqa": ocrqa,
        "source_file": f"s3://115-canonical-processed-final/{source_key}",
        "source_bucket": "115-canonical-processed-final",
        "source_key": source_key,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


class TestParseSourceKey:
    def test_typical_shape(self):
        assert corpus_select._parse_source_key(
            "langident/langident-lid-ensemble_multilingual_v2-0-2/BCUL/ABal/ABal-1922.jsonl.bz2"
        ) == ("BCUL", "ABal", "ABal-1922.jsonl.bz2")

    def test_rejects_short_key(self):
        with pytest.raises(ValueError):
            corpus_select._parse_source_key("foo")


class TestRebuiltKeyFor:
    def test_shape(self):
        assert (
            corpus_select._rebuilt_key_for("BNL", "actionfem", 1927)
            == "BNL/actionfem/actionfem-1927.jsonl.bz2"
        )


class TestSelectionConfigCharThreshold:
    def test_per_language_threshold(self):
        cfg = corpus_select.SelectionConfig(
            input_path=Path("/tmp/x"),
            output_path=Path("/tmp/y"),
            min_tokens=4000,
        )
        assert cfg.char_threshold("fr") == 18000
        assert cfg.char_threshold("de") == 14000


class TestSelectCorpus:
    def test_filters_and_samples_per_language(self, tmp_path: Path):
        records = [
            _agg_record(ci_id=f"fr-keep-{i}", lg="fr", provider="LeTemps", year=1900 + i)
            for i in range(5)
        ]
        records += [
            _agg_record(ci_id=f"de-keep-{i}", lg="de", provider="SNL", year=1910 + i, len_chars=15000)
            for i in range(3)
        ]
        records += [
            _agg_record(ci_id="drop-page", tp="page"),
            _agg_record(ci_id="drop-low-ocr", ocrqa=0.5),
            _agg_record(ci_id="drop-short", len_chars=1000),
            _agg_record(ci_id="drop-bad-year", year=1700),
            _agg_record(ci_id="drop-other-lang", lg="en", provider="BL"),
            _agg_record(ci_id="drop-other-provider", lg="fr", provider="BCUL"),
        ]
        input_path = tmp_path / "agg.jsonl"
        output_path = tmp_path / "manifest.jsonl"
        _write_jsonl(input_path, records)

        cfg = corpus_select.SelectionConfig(
            input_path=input_path,
            output_path=output_path,
            n_per_lg=10,
        )
        manifest, stats = corpus_select.select_corpus(cfg)

        kept_ids = {e.ci_id for e in manifest}
        assert kept_ids == {f"fr-keep-{i}" for i in range(5)} | {f"de-keep-{i}" for i in range(3)}
        assert stats.eligible_per_lg == {"fr": 5, "de": 3}
        assert stats.sampled_per_lg == {"fr": 5, "de": 3}
        assert stats.not_article == 1
        assert stats.ocrqa_below_min == 1
        assert stats.too_short == 1
        assert stats.year_out_of_window == 1
        assert stats.lg_out_of_scope == 1
        assert stats.provider_out_of_scope == 1

    def test_sample_is_deterministic_across_runs(self, tmp_path: Path):
        records = [
            _agg_record(ci_id=f"fr-{i}", lg="fr", provider="LeTemps", year=1900 + (i % 50))
            for i in range(50)
        ]
        input_path = tmp_path / "agg.jsonl"
        _write_jsonl(input_path, records)
        cfg = corpus_select.SelectionConfig(
            input_path=input_path,
            output_path=tmp_path / "out.jsonl",
            n_per_lg=10,
            seed=123,
        )
        manifest_a, _ = corpus_select.select_corpus(cfg)
        manifest_b, _ = corpus_select.select_corpus(cfg)
        assert [e.ci_id for e in manifest_a] == [e.ci_id for e in manifest_b]

    def test_increasing_n_per_lg_extends_previous_sample(self, tmp_path: Path):
        # Stable-seeded-shuffle property: bumping --n-per-lg from N to M (M>N)
        # must extend the previous sample with M-N new docs, NOT reshuffle.
        # This is the property random.sample(pool, k) lacks.
        records = []
        for lg, provider in [("fr", "LeTemps"), ("de", "SNL")]:
            len_chars = 20000 if lg == "fr" else 15000
            for i in range(80):
                records.append(
                    _agg_record(
                        ci_id=f"{lg}-{i:03d}",
                        lg=lg,
                        provider=provider,
                        year=1900 + (i % 50),
                        len_chars=len_chars,
                    )
                )
        input_path = tmp_path / "agg.jsonl"
        _write_jsonl(input_path, records)

        def ids_for(n: int) -> dict[str, set[str]]:
            cfg = corpus_select.SelectionConfig(
                input_path=input_path,
                output_path=tmp_path / "out.jsonl",
                n_per_lg=n,
                seed=42,
            )
            manifest, _ = corpus_select.select_corpus(cfg)
            out: dict[str, set[str]] = {"fr": set(), "de": set()}
            for e in manifest:
                out[e.lg].add(e.ci_id)
            return out

        small = ids_for(10)
        medium = ids_for(25)
        large = ids_for(50)
        for lg in ("fr", "de"):
            assert len(small[lg]) == 10
            assert len(medium[lg]) == 25
            assert len(large[lg]) == 50
            assert small[lg].issubset(medium[lg]), f"{lg}: 10-doc sample not ⊂ 25-doc sample"
            assert medium[lg].issubset(large[lg]), f"{lg}: 25-doc sample not ⊂ 50-doc sample"

    def test_sample_size_capped_to_pool(self, tmp_path: Path):
        records = [
            _agg_record(ci_id=f"fr-{i}", lg="fr", provider="LeTemps", year=1900 + i)
            for i in range(3)
        ]
        input_path = tmp_path / "agg.jsonl"
        _write_jsonl(input_path, records)
        cfg = corpus_select.SelectionConfig(
            input_path=input_path,
            output_path=tmp_path / "out.jsonl",
            n_per_lg=100,
        )
        manifest, stats = corpus_select.select_corpus(cfg)
        assert len(manifest) == 3
        assert stats.sampled_per_lg["fr"] == 3

    def test_per_language_char_threshold_applies(self, tmp_path: Path):
        # de threshold at min_tokens=4000 is 14000 chars (3.5 * 4000); a 15000-char
        # de article passes but a 13000-char one does not. The same de article
        # would NOT pass under fr's 18000-char threshold, so the per-lg knob
        # must be honoured.
        records = [
            _agg_record(ci_id="de-pass", lg="de", provider="SNL", len_chars=15000),
            _agg_record(ci_id="de-fail", lg="de", provider="SNL", len_chars=13000),
        ]
        input_path = tmp_path / "agg.jsonl"
        _write_jsonl(input_path, records)
        cfg = corpus_select.SelectionConfig(
            input_path=input_path,
            output_path=tmp_path / "out.jsonl",
            n_per_lg=10,
        )
        manifest, _ = corpus_select.select_corpus(cfg)
        assert {e.ci_id for e in manifest} == {"de-pass"}

    def test_provider_wildcard_accepts_any_provider(self, tmp_path: Path):
        # ``providers={"fr": None}`` (or empty tuple) disables the provider
        # filter for that language — any provider in the input is accepted.
        records = [
            _agg_record(ci_id="fr-letemps", lg="fr", provider="LeTemps"),
            _agg_record(ci_id="fr-bcul", lg="fr", provider="BCUL"),
            _agg_record(ci_id="fr-rando", lg="fr", provider="WhateverPress"),
        ]
        input_path = tmp_path / "agg.jsonl"
        _write_jsonl(input_path, records)
        cfg = corpus_select.SelectionConfig(
            input_path=input_path,
            output_path=tmp_path / "out.jsonl",
            languages=("fr",),
            providers={"fr": None},
            n_per_lg=10,
        )
        manifest, stats = corpus_select.select_corpus(cfg)
        assert {e.ci_id for e in manifest} == {
            "fr-letemps",
            "fr-bcul",
            "fr-rando",
        }
        assert stats.provider_out_of_scope == 0

    def test_provider_empty_tuple_also_means_all(self, tmp_path: Path):
        records = [_agg_record(ci_id="fr-rando", lg="fr", provider="WhateverPress")]
        input_path = tmp_path / "agg.jsonl"
        _write_jsonl(input_path, records)
        cfg = corpus_select.SelectionConfig(
            input_path=input_path,
            output_path=tmp_path / "out.jsonl",
            languages=("fr",),
            providers={"fr": ()},
            n_per_lg=10,
        )
        manifest, _ = corpus_select.select_corpus(cfg)
        assert {e.ci_id for e in manifest} == {"fr-rando"}

    def test_rebuilt_key_resolution(self, tmp_path: Path):
        records = [_agg_record(ci_id="fr-1", lg="fr", provider="LeTemps", alias="LET", year=1925)]
        input_path = tmp_path / "agg.jsonl"
        _write_jsonl(input_path, records)
        cfg = corpus_select.SelectionConfig(
            input_path=input_path, output_path=tmp_path / "out.jsonl"
        )
        manifest, _ = corpus_select.select_corpus(cfg)
        assert manifest[0].rebuilt_bucket == "122-rebuilt-final"
        assert manifest[0].rebuilt_key == "LeTemps/LET/LET-1925.jsonl.bz2"


class TestWriteManifest:
    def test_round_trip(self, tmp_path: Path):
        records = [
            _agg_record(ci_id=f"fr-{i}", lg="fr", provider="LeTemps", year=1920 + i)
            for i in range(2)
        ]
        input_path = tmp_path / "agg.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_jsonl(input_path, records)
        cfg = corpus_select.SelectionConfig(
            input_path=input_path, output_path=output_path, n_per_lg=10
        )
        manifest, _ = corpus_select.select_corpus(cfg)
        corpus_select.write_manifest(manifest, output_path)
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert set(first.keys()) >= {
            "ci_id",
            "lg",
            "year",
            "len_chars",
            "ocrqa",
            "provider",
            "alias",
            "rebuilt_bucket",
            "rebuilt_key",
        }


class TestParseProviderOverrides:
    def test_basic(self):
        assert corpus_select._parse_provider_overrides(["fr=LeTemps,BNF", "de=NZZ"]) == {
            "fr": ("LeTemps", "BNF"),
            "de": ("NZZ",),
        }

    def test_rejects_missing_equals(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            corpus_select._parse_provider_overrides(["frLeTemps"])


class TestCli:
    def test_main_writes_manifest(self, tmp_path: Path):
        records = [
            _agg_record(ci_id=f"fr-{i}", lg="fr", provider="LeTemps", year=1920 + i)
            for i in range(3)
        ]
        input_path = tmp_path / "agg.jsonl"
        output_path = tmp_path / "manifest.jsonl"
        _write_jsonl(input_path, records)

        rc = corpus_select.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--languages",
                "fr",
                "--n-per-lg",
                "5",
            ]
        )
        assert rc == 0
        assert output_path.exists()
        assert len(output_path.read_text().strip().split("\n")) == 3
