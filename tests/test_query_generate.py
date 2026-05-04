from __future__ import annotations

import asyncio
import bz2
import dataclasses
from pathlib import Path
from typing import Any

import orjson
import pytest

from impresso_text_embedder.research import query_generate as qg


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


class TestBucketRanges:
    def test_three_equal_thirds(self):
        assert qg.bucket_ranges(30) == ((0, 10), (10, 20), (20, 30))

    def test_remainder_lands_in_tail(self):
        ranges = qg.bucket_ranges(31)
        assert ranges[0] == (0, 10)
        assert ranges[1] == (10, 20)
        assert ranges[2] == (20, 31)

    def test_empty_doc(self):
        assert qg.bucket_ranges(0) == ((0, 0), (0, 0), (0, 0))

    def test_buckets_are_contiguous_and_cover_doc(self):
        for total in (1, 7, 100, 17_000):
            ranges = qg.bucket_ranges(total)
            assert ranges[0][0] == 0
            assert ranges[2][1] == total
            assert ranges[0][1] == ranges[1][0]
            assert ranges[1][1] == ranges[2][0]

    def test_n_parameter_generalizes_bucket_count(self):
        # Quintile split — five contiguous buckets, last absorbs remainder.
        ranges = qg.bucket_ranges(53, n=5)
        assert len(ranges) == 5
        assert ranges[0] == (0, 10)
        assert ranges[-1] == (40, 53)
        # All buckets are contiguous.
        for a, b in zip(ranges, ranges[1:]):
            assert a[1] == b[0]

    def test_n_must_be_positive(self):
        with pytest.raises(ValueError):
            qg.bucket_ranges(100, n=0)

    def test_n_buckets_for_empty_doc_returns_n_zero_slots(self):
        assert qg.bucket_ranges(0, n=5) == ((0, 0),) * 5


# ---------------------------------------------------------------------------
# Verbatim verification
# ---------------------------------------------------------------------------


class TestVerifyReferences:
    def test_in_bucket_match_kept(self):
        ft = "head text. mid text with target. tail text."
        bucket_range = (10, 33)
        verified, nf, oob = qg.verify_references(ft, ["target"], bucket_range)
        assert nf == 0 and oob == 0
        assert len(verified) == 1
        assert verified[0].text == "target"
        assert verified[0].char_start == ft.index("target")
        assert verified[0].char_end == ft.index("target") + len("target")
        assert ft[verified[0].char_start : verified[0].char_end] == "target"

    def test_out_of_bucket_match_rejected(self):
        ft = "head text. mid text. tail target."
        verified, nf, oob = qg.verify_references(ft, ["target"], (10, 20))
        assert verified == []
        assert oob == 1 and nf == 0

    def test_first_in_bucket_wins_over_earlier_out_of_bucket(self):
        ft = "target head. target mid. target tail."
        verified, nf, oob = qg.verify_references(ft, ["target"], (12, 24))
        assert nf == 0 and oob == 0
        assert len(verified) == 1
        assert verified[0].char_start == 13

    def test_not_found_anywhere(self):
        ft = "head text mid text tail text"
        verified, nf, oob = qg.verify_references(ft, ["absent"], (0, 28))
        assert verified == []
        assert nf == 1 and oob == 0

    def test_empty_and_non_string_refs_count_as_not_found(self):
        ft = "head text"
        verified, nf, oob = qg.verify_references(ft, ["", "   "], (0, 9))
        assert verified == []
        assert nf == 2 and oob == 0


# ---------------------------------------------------------------------------
# Pydantic schema — ensure the contract LangChain parses against is sound
# ---------------------------------------------------------------------------


class TestQueryOutput:
    def test_accepts_minimal_payload(self):
        out = qg.QueryOutput(query="Q?", references=["alpha"])
        assert out.query == "Q?"
        assert out.references == ["alpha"]

    def test_references_default_empty(self):
        out = qg.QueryOutput(query="Q?")
        assert out.references == []

    def test_query_is_required(self):
        with pytest.raises(Exception):  # pydantic.ValidationError
            qg.QueryOutput()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Prompts — language and bucket label propagation
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_question_prompt_mentions_question(self):
        prompt = qg.build_system_prompt("question", "fr")
        assert "question" in prompt.lower()
        assert "French" in prompt

    def test_topical_phrase_prompt_mentions_phrase_and_word_count(self):
        prompt = qg.build_system_prompt("topical-phrase", "de")
        assert "phrase" in prompt.lower() or "keyword" in prompt.lower()
        assert "German" in prompt

    def test_unknown_query_type_raises(self):
        with pytest.raises(ValueError):
            qg.build_system_prompt("summary", "fr")  # type: ignore[arg-type]


class TestUserMessage:
    def test_includes_focus_region_and_full_article(self):
        record = qg.CorpusRecord(ci_id="x", lg="fr", ft="aaa bbb ccc")
        msg = qg.build_user_message(record, "head", "aaa")
        assert "FOCUS REGION (head" in msg
        assert "aaa" in msg
        assert "FULL ARTICLE" in msg
        assert "aaa bbb ccc" in msg

    def test_drops_full_article_when_doc_exceeds_char_limit(self):
        ft = "x" * (qg._INCLUDE_FULL_ARTICLE_CHAR_LIMIT + 1)
        record = qg.CorpusRecord(ci_id="x", lg="fr", ft=ft)
        msg = qg.build_user_message(record, "head", "x" * 100)
        assert "FOCUS REGION (head" in msg
        assert "FULL ARTICLE" not in msg


# ---------------------------------------------------------------------------
# Job planning
# ---------------------------------------------------------------------------


class TestPlanJobs:
    def test_six_jobs_per_record(self):
        records = [qg.CorpusRecord(ci_id="r1", lg="fr", ft="abc def ghi jkl mno pqr stu vwx yz")]
        cfg = qg.GenerationConfig(api_key="k")
        jobs = qg._plan_jobs(records, cfg)
        assert len(jobs) == 6
        assert [j.bucket_idx for j in jobs] == [0, 0, 1, 1, 2, 2]
        assert [j.query_type for j in jobs] == [
            "question",
            "topical-phrase",
            "question",
            "topical-phrase",
            "question",
            "topical-phrase",
        ]

    def test_empty_doc_yields_no_jobs(self):
        records = [qg.CorpusRecord(ci_id="empty", lg="fr", ft="")]
        cfg = qg.GenerationConfig(api_key="k")
        assert qg._plan_jobs(records, cfg) == []

    def test_quintile_buckets_via_position_buckets(self):
        """Move 1 — bucket count is now config-driven via position_buckets."""
        records = [qg.CorpusRecord(ci_id="r1", lg="fr", ft="x" * 50)]
        cfg = qg.GenerationConfig(
            api_key="k",
            position_buckets=("q0", "q1", "q2", "q3", "q4"),
        )
        jobs = qg._plan_jobs(records, cfg)
        # 5 buckets × 2 query_types × 1 sample = 10 jobs
        assert len(jobs) == 10
        assert sorted({j.bucket_label for j in jobs}) == ["q0", "q1", "q2", "q3", "q4"]
        # Buckets contiguous and non-empty
        ranges = sorted({(j.bucket_range, j.bucket_label) for j in jobs})
        assert ranges[0][0] == (0, 10)
        assert ranges[-1][0] == (40, 50)

    def test_queries_per_bucket_multiplies_jobs(self):
        """Move 3 — queries_per_bucket=N emits N samples per (record, bucket, type)."""
        records = [qg.CorpusRecord(ci_id="r1", lg="fr", ft="x" * 30)]
        cfg = qg.GenerationConfig(api_key="k", queries_per_bucket=3)
        jobs = qg._plan_jobs(records, cfg)
        # 3 buckets × 2 query_types × 3 samples = 18 jobs
        assert len(jobs) == 18
        # Sample indices cycle 0..2 within each (bucket, query_type) cell
        head_q = [j for j in jobs if j.bucket_label == "head" and j.query_type == "question"]
        assert [j.sample_idx for j in head_q] == [0, 1, 2]

    def test_default_config_uses_three_head_mid_tail_buckets(self):
        """Backward compat — default GenerationConfig matches v1 behaviour."""
        records = [qg.CorpusRecord(ci_id="r", lg="fr", ft="x" * 30)]
        cfg = qg.GenerationConfig(api_key="k")
        jobs = qg._plan_jobs(records, cfg)
        assert len(jobs) == 6
        assert [j.bucket_label for j in jobs] == [
            "head", "head", "mid", "mid", "tail", "tail",
        ]


# ---------------------------------------------------------------------------
# generate_one — end-to-end with a stub LangChain runnable
# ---------------------------------------------------------------------------


class _StubLLM:
    """Stub matching the langchain runnable surface used by generate_one.

    Each entry in ``responses`` is either a :class:`qg.QueryOutput`
    (returned by the next ``ainvoke``) or an :class:`Exception`
    instance (raised by the next ``ainvoke``). Real LangChain
    ``with_retry`` would have already exhausted its budget by the time
    an exception bubbles to ``generate_one``, so the test stub raises
    immediately.
    """

    def __init__(self, responses: list[qg.QueryOutput | Exception] | None = None) -> None:
        self.calls: list[Any] = []
        self._responses = list(responses or [])

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("stub: no scripted response left")
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _make_record(ft: str, *, ci_id: str = "test-ci", lg: str = "fr") -> qg.CorpusRecord:
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
    return qg.GenerationConfig(api_key="test-key", max_parallel=1)


# 53-char ft → buckets [0,17) head, [17,34) mid, [34,53) tail.
# "alpha" only in head, "beta" only in mid, "gamma" only in tail.
_FT_TRIPLE = "alpha alpha alpha. beta beta beta. gamma gamma gamma."
assert len(_FT_TRIPLE) == 53


class TestGenerateOne:
    def test_happy_path(self, cfg):
        llm = _StubLLM([qg.QueryOutput(query="Was?", references=["beta"])])
        job = _make_job(_make_record(_FT_TRIPLE), bucket_idx=1)
        result = asyncio.run(qg.generate_one(job, cfg, llm))
        assert result.error_kind == "ok"
        assert result.query is not None
        q = result.query
        assert q.position_bucket == "mid"
        assert q.lg == "fr"
        assert q.query_text == "Was?"
        assert len(q.references) == 1
        ref = q.references[0]
        assert ref.text == "beta"
        assert _FT_TRIPLE[ref.char_start : ref.char_end] == "beta"
        lo, hi = q.position_chars
        assert lo <= ref.char_start < hi

    def test_drops_when_ref_not_in_bucket(self, cfg):
        llm = _StubLLM([qg.QueryOutput(query="Q", references=["alpha"])])
        job = _make_job(_make_record(_FT_TRIPLE), bucket_idx=1)
        result = asyncio.run(qg.generate_one(job, cfg, llm))
        assert result.query is None
        assert result.error_kind == "no_refs"
        assert result.refs_out_of_bucket == 1
        assert result.refs_not_found == 0

    def test_drops_when_ref_absent(self, cfg):
        llm = _StubLLM([qg.QueryOutput(query="Q", references=["fictional"])])
        job = _make_job(_make_record(_FT_TRIPLE), bucket_idx=0)
        result = asyncio.run(qg.generate_one(job, cfg, llm))
        assert result.query is None
        assert result.error_kind == "no_refs"
        assert result.refs_not_found == 1
        assert result.refs_out_of_bucket == 0

    def test_records_api_error(self, cfg):
        # Empty stub → raises on first call (simulates a transient HTTP
        # failure that survived langchain's retry budget).
        llm = _StubLLM([])
        job = _make_job(_make_record(_FT_TRIPLE), bucket_idx=0)
        result = asyncio.run(qg.generate_one(job, cfg, llm))
        assert result.query is None
        assert result.error_kind == "api"

    def test_query_id_format(self, cfg):
        llm = _StubLLM([qg.QueryOutput(query="Q", references=["beta"])])
        job = _make_job(
            _make_record(_FT_TRIPLE, ci_id="abc-123"),
            bucket_idx=1,
            query_type="topical-phrase",
        )
        result = asyncio.run(qg.generate_one(job, cfg, llm))
        assert result.query is not None
        # query_id always carries the __NN sample-index suffix; with
        # queries_per_bucket=1 this is always __00.
        assert result.query.query_id == "abc-123__mid__topical-phrase__00"

    def test_query_id_includes_sample_idx_when_multiple_per_bucket(self, cfg):
        llm = _StubLLM([qg.QueryOutput(query="Q", references=["beta"])])
        job = _make_job(
            _make_record(_FT_TRIPLE, ci_id="abc-123"),
            bucket_idx=1,
            query_type="question",
            sample_idx=2,
        )
        result = asyncio.run(qg.generate_one(job, cfg, llm))
        assert result.query is not None
        assert result.query.query_id == "abc-123__mid__question__02"


# ---------------------------------------------------------------------------
# generate_queries — end-to-end with a stub runnable
# ---------------------------------------------------------------------------


class TestGenerateQueries:
    def test_six_queries_per_record_in_deterministic_order(self, cfg):
        llm = _StubLLM(
            [
                qg.QueryOutput(query="Q-head-q", references=["alpha"]),
                qg.QueryOutput(query="Q-head-tp", references=["alpha"]),
                qg.QueryOutput(query="Q-mid-q", references=["beta"]),
                qg.QueryOutput(query="Q-mid-tp", references=["beta"]),
                qg.QueryOutput(query="Q-tail-q", references=["gamma"]),
                qg.QueryOutput(query="Q-tail-tp", references=["gamma"]),
            ]
        )
        records = [_make_record(_FT_TRIPLE, ci_id="r1")]
        cfg_serial = dataclasses.replace(cfg, max_parallel=1)
        queries, stats = asyncio.run(qg.generate_queries(records, cfg_serial, llm))
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
        assert stats.by_bucket == {"head": 2, "mid": 2, "tail": 2}
        assert stats.by_query_type == {"question": 3, "topical-phrase": 3}
        assert stats.by_lg == {"fr": 6}

    def test_drops_propagate_into_stats(self, cfg):
        llm = _StubLLM([qg.QueryOutput(query="Q", references=["zzz"])] * 6)
        cfg_serial = dataclasses.replace(cfg, max_parallel=1)
        queries, stats = asyncio.run(
            qg.generate_queries([_make_record(_FT_TRIPLE)], cfg_serial, llm)
        )
        assert queries == []
        assert stats.queries_kept == 0
        assert stats.no_refs_returned == 6
        assert stats.refs_not_found == 6

    def test_empty_corpus(self, cfg):
        queries, stats = asyncio.run(qg.generate_queries([], cfg, _StubLLM()))
        assert queries == []
        assert stats.attempts == 0


# ---------------------------------------------------------------------------
# I/O — corpus shard read + queries write round-trip
# ---------------------------------------------------------------------------


def _write_corpus_shard(path: Path, records: list[dict]) -> None:
    with bz2.open(path, "wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))


class TestReadCorpusShard:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl.bz2"
        _write_corpus_shard(
            path,
            [
                {"ci_id": "a", "lg": "fr", "ft": "alpha"},
                {"ci_id": "b", "lg": "de", "ft": "bravo"},
            ],
        )
        loaded = qg.read_corpus_shard(path)
        assert [r.ci_id for r in loaded] == ["a", "b"]
        assert loaded[1].lg == "de"
        assert loaded[1].ft == "bravo"

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl.bz2"
        with bz2.open(path, "wb") as fh:
            fh.write(orjson.dumps({"ci_id": "a", "lg": "fr", "ft": "x"}, option=orjson.OPT_APPEND_NEWLINE))
            fh.write(b"\n")
            fh.write(orjson.dumps({"ci_id": "b", "lg": "de", "ft": "y"}, option=orjson.OPT_APPEND_NEWLINE))
        loaded = qg.read_corpus_shard(path)
        assert [r.ci_id for r in loaded] == ["a", "b"]


class TestWriteQueries:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "queries.jsonl.bz2"
        q = qg.Query(
            query_id="r1__head__question",
            ci_id="r1",
            lg="fr",
            query_text="Q?",
            query_type="question",
            references=(qg.Reference(text="alpha", char_start=0, char_end=5),),
            position_bucket="head",
            position_chars=(0, 100),
            gen_model="model",
            gen_endpoint="endpoint",
            ts="2026-04-29T12:00:00Z",
        )
        qg.write_queries([q], path)
        with bz2.open(path, "rb") as fh:
            line = fh.readline()
        loaded = orjson.loads(line)
        assert loaded["query_id"] == "r1__head__question"
        assert loaded["references"] == [{"text": "alpha", "char_start": 0, "char_end": 5}]
        assert loaded["position_chars"] == [0, 100]


# ---------------------------------------------------------------------------
# CLI plumbing — argparse contract
# ---------------------------------------------------------------------------


class TestConfigFromArgs:
    """``config_from_args`` now requires a ``study_cfg``; build one from
    the frozen ``study-v1.yaml`` fixture so we exercise real merge logic.
    """

    @pytest.fixture
    def study_cfg(self):
        from impresso_text_embedder.research.study_config import load_study_config

        return load_study_config(
            Path(__file__).parent.parent / "configs/research/study-v1.yaml"
        )

    def test_requires_api_key(self, monkeypatch, study_cfg):
        monkeypatch.delenv("RCP_API_KEY", raising=False)
        args = qg.build_parser().parse_args(["--config", "x"])
        with pytest.raises(SystemExit, match="RCP_API_KEY"):
            qg.config_from_args(args, study_cfg)

    def test_uses_env_when_set(self, monkeypatch, study_cfg):
        monkeypatch.setenv("RCP_API_KEY", "from-env")
        args = qg.build_parser().parse_args(["--config", "x"])
        cfg = qg.config_from_args(args, study_cfg)
        assert cfg.api_key == "from-env"

    def test_cli_override_wins(self, monkeypatch, study_cfg):
        monkeypatch.setenv("RCP_API_KEY", "from-env")
        args = qg.build_parser().parse_args(["--config", "x", "--api-key", "from-cli"])
        cfg = qg.config_from_args(args, study_cfg)
        assert cfg.api_key == "from-cli"

    def test_study_cfg_supplies_query_generation_defaults(
        self, monkeypatch, study_cfg
    ):
        """When the CLI flag matches the argparse default, the YAML wins."""
        monkeypatch.setenv("RCP_API_KEY", "k")
        args = qg.build_parser().parse_args(["--config", "x"])
        cfg = qg.config_from_args(args, study_cfg)
        # base.yaml pins these; assert the study config bled through.
        assert cfg.model == study_cfg.query_generation.model
        assert cfg.position_buckets == tuple(study_cfg.query_generation.position_buckets)
        assert cfg.study_name == study_cfg.study.name
        assert cfg.study_config_sha == study_cfg.config_sha
