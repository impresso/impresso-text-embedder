"""Unit tests for the CaaS variant of query-generation.

The local-inference path lives in
:mod:`impresso_text_embedder.research.query_generate_local`. These
tests exercise the surfaces that don't need real model weights:

- ``parse_query_output`` — fence stripping + JSON parsing + Pydantic
  validation.
- ``generate_one_local`` and ``generate_queries_local`` — the
  per-job and run-level loops, driven by an injected ``generate_fn``
  closure that stands in for ``generate_completion``.
- ``config_from_args`` — argparse + study-YAML merge wiring.

Real model loading (``load_local_llm`` / ``generate_completion``) is
out of scope here — those run against actual transformers weights on
a GPU host and belong with the functional/GPU-correctness tests, not
the local CPU-only suite.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from impresso_text_embedder.research import query_generate as qg
from impresso_text_embedder.research import query_generate_local as qgl

# ---------------------------------------------------------------------------
# parse_query_output
# ---------------------------------------------------------------------------


class TestParseQueryOutput:
    def test_plain_json(self):
        out = qgl.parse_query_output('{"query": "Q?", "references": ["a", "b"]}')
        assert out is not None
        assert out.query == "Q?"
        assert out.references == ["a", "b"]

    def test_strips_json_fence(self):
        text = '```json\n{"query": "Q?", "references": ["a"]}\n```'
        out = qgl.parse_query_output(text)
        assert out is not None
        assert out.query == "Q?"

    def test_strips_plain_fence(self):
        text = '```\n{"query": "Q?", "references": []}\n```'
        out = qgl.parse_query_output(text)
        assert out is not None
        assert out.references == []

    def test_handles_leading_prose(self):
        text = 'Sure! Here is the JSON:\n{"query": "Q?", "references": ["a"]}'
        out = qgl.parse_query_output(text)
        assert out is not None
        assert out.query == "Q?"

    def test_returns_none_on_garbage(self):
        assert qgl.parse_query_output("not json at all") is None

    def test_returns_none_on_truncated_json(self):
        # Unterminated string — orjson raises JSONDecodeError; we eat it.
        assert qgl.parse_query_output('{"query": "Q?", "references": ["a') is None

    def test_returns_none_on_schema_violation(self):
        # Pydantic rejects a non-string `query`.
        assert qgl.parse_query_output('{"query": 123, "references": []}') is None


# ---------------------------------------------------------------------------
# generate_one_local — synchronous, with stub generate_fn
# ---------------------------------------------------------------------------


_FT_TRIPLE = "alpha alpha alpha. beta beta beta. gamma gamma gamma."
assert len(_FT_TRIPLE) == 53  # buckets [0,17) head, [17,34) mid, [34,53) tail


def _make_record(ft: str = _FT_TRIPLE, *, ci_id: str = "ci", lg: str = "fr") -> qg.CorpusRecord:
    return qg.CorpusRecord(ci_id=ci_id, lg=lg, ft=ft)


def _make_job(
    record: qg.CorpusRecord,
    bucket_idx: int,
    query_type: qg.QueryType = "question",
    sample_idx: int = 0,
    bucket_labels: tuple[str, ...] = qg.DEFAULT_POSITION_BUCKETS,
) -> qg.Job:
    lo, hi = qg.bucket_ranges(len(record.ft), len(bucket_labels))[bucket_idx]
    return qg.Job(
        record=record,
        bucket_idx=bucket_idx,
        query_type=query_type,
        bucket_range=(lo, hi),
        bucket_text=record.ft[lo:hi],
        bucket_label=bucket_labels[bucket_idx],
        sample_idx=sample_idx,
    )


@pytest.fixture
def cfg() -> qg.GenerationConfig:
    return qg.GenerationConfig(model="local-stub", endpoint=qgl.LOCAL_ENDPOINT)


class TestGenerateOneLocal:
    def test_happy_path(self, cfg):
        completions = ['{"query": "Was?", "references": ["beta"]}']

        def gen(messages):
            return completions.pop(0)

        job = _make_job(_make_record(), bucket_idx=1)
        result = qgl.generate_one_local(job, cfg, generate_fn=gen, max_retries=1)
        assert result.error_kind == "ok"
        assert result.query is not None
        q = result.query
        assert q.position_bucket == "mid"
        assert q.gen_endpoint == qgl.LOCAL_ENDPOINT
        assert q.gen_model == "local-stub"
        assert q.references[0].text == "beta"

    def test_retries_on_parse_failure(self, cfg):
        # First call returns garbage, second returns valid JSON.
        completions = ["not json", '{"query": "Q?", "references": ["beta"]}']

        def gen(messages):
            return completions.pop(0)

        job = _make_job(_make_record(), bucket_idx=1)
        result = qgl.generate_one_local(job, cfg, generate_fn=gen, max_retries=3)
        assert result.error_kind == "ok"
        assert result.query is not None
        # Both completions consumed.
        assert completions == []

    def test_gives_up_after_max_retries(self, cfg):
        def gen(messages):
            return "still not json"

        job = _make_job(_make_record(), bucket_idx=1)
        result = qgl.generate_one_local(job, cfg, generate_fn=gen, max_retries=2)
        assert result.query is None
        assert result.error_kind == "api"

    def test_records_api_error(self, cfg):
        def gen(messages):
            raise RuntimeError("CUDA oom")

        job = _make_job(_make_record(), bucket_idx=0)
        result = qgl.generate_one_local(job, cfg, generate_fn=gen, max_retries=3)
        assert result.query is None
        assert result.error_kind == "api"

    def test_drops_when_ref_not_in_bucket(self, cfg):
        def gen(messages):
            # "alpha" is in the head bucket; the job targets mid.
            return '{"query": "Q?", "references": ["alpha"]}'

        job = _make_job(_make_record(), bucket_idx=1)
        result = qgl.generate_one_local(job, cfg, generate_fn=gen, max_retries=1)
        assert result.query is None
        assert result.error_kind == "no_refs"
        assert result.refs_out_of_bucket == 1

    def test_query_id_format(self, cfg):
        def gen(messages):
            return '{"query": "Q", "references": ["beta"]}'

        job = _make_job(
            _make_record(ci_id="abc-123"),
            bucket_idx=1,
            query_type="topical-phrase",
        )
        result = qgl.generate_one_local(job, cfg, generate_fn=gen, max_retries=1)
        assert result.query is not None
        assert result.query.query_id == "abc-123__mid__topical-phrase__00"


# ---------------------------------------------------------------------------
# generate_queries_local — full sync loop
# ---------------------------------------------------------------------------


class TestGenerateQueriesLocal:
    def test_six_queries_per_record_in_deterministic_order(self, cfg):
        completions = [
            '{"query": "Q-head-q", "references": ["alpha"]}',
            '{"query": "Q-head-tp", "references": ["alpha"]}',
            '{"query": "Q-mid-q", "references": ["beta"]}',
            '{"query": "Q-mid-tp", "references": ["beta"]}',
            '{"query": "Q-tail-q", "references": ["gamma"]}',
            '{"query": "Q-tail-tp", "references": ["gamma"]}',
        ]

        def gen(messages):
            return completions.pop(0)

        records = [_make_record(ci_id="r1")]
        queries, stats = qgl.generate_queries_local(
            records, cfg, generate_fn=gen, max_retries=1
        )
        assert len(queries) == 6
        assert [q.position_bucket for q in queries] == [
            "head", "head", "mid", "mid", "tail", "tail",
        ]
        assert [q.query_type for q in queries] == [
            "question", "topical-phrase",
            "question", "topical-phrase",
            "question", "topical-phrase",
        ]
        assert stats.queries_kept == 6
        assert stats.attempts == 6
        assert stats.by_query_type == {"question": 3, "topical-phrase": 3}

    def test_empty_corpus(self, cfg):
        def gen(messages):
            raise AssertionError("should not be called")

        queries, stats = qgl.generate_queries_local(
            [], cfg, generate_fn=gen, max_retries=1
        )
        assert queries == []
        assert stats.attempts == 0


# ---------------------------------------------------------------------------
# CLI plumbing — argparse + study-YAML merge
# ---------------------------------------------------------------------------


class TestConfigFromArgs:
    @pytest.fixture
    def study_cfg(self):
        from impresso_text_embedder.research.study_config import load_study_config

        return load_study_config(
            Path(__file__).parent.parent / "configs/research/study-v1.yaml"
        )

    def test_uses_study_defaults(self, study_cfg):
        args = qgl.build_parser().parse_args(["--config", "x"])
        gen, knobs = qgl.config_from_args(args, study_cfg)
        # base.yaml pins the model; the YAML should bleed through when
        # the CLI flag matches its argparse default.
        assert gen.model == study_cfg.query_generation.model
        assert gen.position_buckets == tuple(
            study_cfg.query_generation.position_buckets
        )
        assert gen.study_name == study_cfg.study.name
        assert gen.study_config_sha == study_cfg.config_sha
        assert gen.endpoint == qgl.LOCAL_ENDPOINT
        # Local-only knobs at their defaults.
        assert knobs.dtype == qgl.DEFAULT_DTYPE
        assert knobs.attention == qgl.DEFAULT_ATTENTION
        assert knobs.max_retries == qgl.DEFAULT_MAX_RETRIES

    def test_cli_overrides_win(self, study_cfg):
        args = qgl.build_parser().parse_args([
            "--config", "x",
            "--model", "some-other-model",
            "--dtype", "fp16",
            "--attention", "flash_attention_2",
            "--max-retries", "7",
            "--temperature", "0.0",
        ])
        gen, knobs = qgl.config_from_args(args, study_cfg)
        assert gen.model == "some-other-model"
        assert gen.temperature == 0.0
        assert knobs.dtype == "fp16"
        assert knobs.attention == "flash_attention_2"
        assert knobs.max_retries == 7

    def test_does_not_require_api_key(self, monkeypatch, study_cfg):
        # Unlike the AIaaS path, the CaaS path doesn't need RCP_API_KEY.
        monkeypatch.delenv("RCP_API_KEY", raising=False)
        args = qgl.build_parser().parse_args(["--config", "x"])
        # Should not raise.
        gen, _ = qgl.config_from_args(args, study_cfg)
        assert gen.api_key == ""


# ---------------------------------------------------------------------------
# Sanity: the CaaS path emits the same Query schema as the AIaaS path
# ---------------------------------------------------------------------------


class TestSchemaParity:
    """The eval downstream cannot tell the two scripts apart on schema."""

    def test_query_dataclass_fields_match(self, cfg):
        completions = ['{"query": "Q?", "references": ["beta"]}']

        def gen(messages):
            return completions.pop(0)

        job = _make_job(_make_record(), bucket_idx=1)
        result = qgl.generate_one_local(job, cfg, generate_fn=gen, max_retries=1)
        assert result.query is not None
        # Same dataclass, same field set as the AIaaS path.
        assert {f.name for f in dataclasses.fields(result.query)} == {
            f.name for f in dataclasses.fields(qg.Query)
        }
