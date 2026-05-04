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

        def gen(batch):
            return [completions.pop(0) for _ in batch]

        records = [_make_record(ci_id="r1")]
        queries, stats = qgl.generate_queries_local(
            records, cfg, generate_fn=gen, batch_size=1, max_retries=1
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

    def test_batched_generation_groups_jobs(self, cfg):
        """batch_size>1: generate_fn receives N messages per call."""
        completions = [
            '{"query": "Q-head-q", "references": ["alpha"]}',
            '{"query": "Q-head-tp", "references": ["alpha"]}',
            '{"query": "Q-mid-q", "references": ["beta"]}',
            '{"query": "Q-mid-tp", "references": ["beta"]}',
            '{"query": "Q-tail-q", "references": ["gamma"]}',
            '{"query": "Q-tail-tp", "references": ["gamma"]}',
        ]
        batch_sizes_seen: list[int] = []

        def gen(batch):
            batch_sizes_seen.append(len(batch))
            return [completions.pop(0) for _ in batch]

        records = [_make_record(ci_id="r1")]
        queries, stats = qgl.generate_queries_local(
            records, cfg, generate_fn=gen, batch_size=4, max_retries=1
        )
        # 6 jobs at batch_size=4 → batches of [4, 2].
        assert batch_sizes_seen == [4, 2]
        # Output identical to the single-stream path: same Query records,
        # same deterministic order — batching is purely a performance
        # knob, not a semantics one.
        assert len(queries) == 6
        assert stats.queries_kept == 6
        assert stats.attempts == 6

    def test_batched_parse_failure_falls_back_to_single_job_retry(self, cfg):
        """A bad completion in the batch retries that prompt alone, not the whole batch."""
        # 6 jobs (3 buckets × 2 query_types) at batch_size=2 → 3 batched
        # calls. The first batch's row 1 emits garbage; that single
        # prompt is retried as a single-job call (NOT the whole batch).
        # All subsequent batched rows parse cleanly. Expected gen() call
        # sizes: [2 (batch 1), 1 (single retry), 2 (batch 2), 2 (batch 3)].
        responses = [
            # batch 1: head-question + head-topical-phrase. Row 1 fails.
            ['{"query": "Q-head-q", "references": ["alpha"]}', "not json"],
            # single-job retry of head-topical-phrase: now valid.
            ['{"query": "Q-head-tp", "references": ["alpha"]}'],
            # batch 2: mid-question + mid-topical-phrase, both valid.
            ['{"query": "Q-mid-q", "references": ["beta"]}',
             '{"query": "Q-mid-tp", "references": ["beta"]}'],
            # batch 3: tail-question + tail-topical-phrase, both valid.
            ['{"query": "Q-tail-q", "references": ["gamma"]}',
             '{"query": "Q-tail-tp", "references": ["gamma"]}'],
        ]
        sizes_seen: list[int] = []

        def gen(batch):
            sizes_seen.append(len(batch))
            return responses.pop(0)

        records = [_make_record(ci_id="r1")]
        queries, stats = qgl.generate_queries_local(
            records, cfg, generate_fn=gen, batch_size=2, max_retries=2
        )
        # Plumbing: one batched call per pair, one single-job call for
        # the retry. The single-job retry MUST be size=1, not size=2 —
        # we don't waste compute re-running the good prompts.
        assert sizes_seen == [2, 1, 2, 2]
        # All 6 jobs eventually produced queries.
        assert stats.queries_kept == 6
        assert stats.attempts == 6
        assert stats.api_errors == 0

    def test_whole_batch_failure_marks_all_jobs_api_error(self, cfg):
        # Non-OOM error (e.g. broken model, tokenizer mismatch) — no
        # single-job fallback because retrying one at a time wouldn't
        # change the outcome. All jobs in the batch get tagged api.
        def gen(batch):
            raise RuntimeError("model is broken")

        records = [_make_record(ci_id="r1")]
        queries, stats = qgl.generate_queries_local(
            records, cfg, generate_fn=gen, batch_size=4, max_retries=1
        )
        # 3 buckets × 2 query_types × 1 sample_idx = 6 jobs, all failed.
        assert queries == []
        assert stats.attempts == 6
        assert stats.api_errors == 6

    def test_oom_falls_back_to_single_job_for_the_batch(self, cfg):
        """OOM at batch=N retries the same prompts at batch=1, recovering most jobs."""
        # First call (batch=4) raises CUDA OOM; subsequent calls are
        # size-1 retries of the same 4 prompts. Then batch 2 (size=2 —
        # tail of 6 jobs) succeeds normally.
        sizes_seen: list[int] = []
        oom_calls_remaining = [True]

        def gen(batch):
            sizes_seen.append(len(batch))
            if oom_calls_remaining[0] and len(batch) == 4:
                oom_calls_remaining[0] = False
                # Mimic the torch.cuda.OutOfMemoryError string shape.
                raise RuntimeError(
                    "CUDA out of memory. Tried to allocate 1.80 GiB. "
                    "GPU 0 has a total capacity of 79.18 GiB ..."
                )
            # Size-1 retries + the second batch all succeed; reference
            # depends on which job the row maps to. _FT_TRIPLE buckets:
            # head→alpha, mid→beta, tail→gamma. Batch 1 (size 4) =
            # head-q, head-tp, mid-q, mid-tp. Batch 2 (size 2) =
            # tail-q, tail-tp.
            jobs_fr = ["alpha", "alpha", "beta", "beta", "gamma", "gamma"]
            base_idx = sum(s for s in sizes_seen[:-1] if s != 4) - (
                # don't count the size=4 OOM in the offset
                0
            )
            ref = jobs_fr[base_idx % 6]
            return [
                f'{{"query": "Q", "references": ["{ref}"]}}'
                for _ in batch
            ]

        records = [_make_record(ci_id="r1")]
        queries, stats = qgl.generate_queries_local(
            records, cfg, generate_fn=gen, batch_size=4, max_retries=1
        )
        # Call shape: [4 (OOM), 1, 1, 1, 1 (single-job fallback), 2 (next batch)].
        assert sizes_seen[0] == 4
        assert sizes_seen[1:5] == [1, 1, 1, 1]
        assert sizes_seen[5] == 2
        # All 6 jobs should have produced queries thanks to the fallback.
        assert stats.attempts == 6
        assert stats.api_errors == 0
        assert stats.queries_kept == 6

    def test_oom_at_batch_size_one_raises_through(self, cfg):
        """Single-job OOM — nothing to fall back to; mark as api error."""
        def gen(batch):
            assert len(batch) == 1
            raise RuntimeError("CUDA out of memory")

        records = [_make_record(ci_id="r1")]
        queries, stats = qgl.generate_queries_local(
            records, cfg, generate_fn=gen, batch_size=1, max_retries=1
        )
        # 6 jobs, all attempted at batch=1, all OOMed.
        assert queries == []
        assert stats.attempts == 6
        assert stats.api_errors == 6

    def test_is_oom_detection(self):
        assert qgl._is_oom(RuntimeError("CUDA out of memory. Tried to ..."))
        assert qgl._is_oom(RuntimeError("cuda oom: ..."))
        assert not qgl._is_oom(RuntimeError("model is broken"))
        # Ducktyped subclass with the right name (covers
        # torch.cuda.OutOfMemoryError without importing torch).
        class OutOfMemoryError(RuntimeError):
            pass
        assert qgl._is_oom(OutOfMemoryError("anything"))


# ---------------------------------------------------------------------------
# Progress logging — periodic ETA + batch-rate snapshot
# ---------------------------------------------------------------------------


import logging  # noqa: E402 — late import keeps the rest of the file lint-flat


class TestFmtDuration:
    def test_seconds(self):
        assert qgl._fmt_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert qgl._fmt_duration(312) == "5m12s"

    def test_hours(self):
        # 1 h 23 m 45 s
        assert qgl._fmt_duration(3600 + 23 * 60 + 45) == "1h23m45s"

    def test_zero(self):
        assert qgl._fmt_duration(0) == "0s"

    def test_negative_clamps_to_zero(self):
        # Defensive: ETA computations can briefly go negative on
        # clock-skew adjustments.
        assert qgl._fmt_duration(-5) == "0s"

    def test_nan_clamps_to_zero(self):
        assert qgl._fmt_duration(float("nan")) == "0s"


class TestProgressLogging:
    def test_log_fires_every_n_batches(self, cfg, caplog):
        # 6 jobs at batch_size=1 → 6 batches. log_every_n=2 → progress
        # lines at batch 2, 4, 6.
        completions = [
            '{"query": "Q-head-q", "references": ["alpha"]}',
            '{"query": "Q-head-tp", "references": ["alpha"]}',
            '{"query": "Q-mid-q", "references": ["beta"]}',
            '{"query": "Q-mid-tp", "references": ["beta"]}',
            '{"query": "Q-tail-q", "references": ["gamma"]}',
            '{"query": "Q-tail-tp", "references": ["gamma"]}',
        ]

        def gen(batch):
            return [completions.pop(0) for _ in batch]

        with caplog.at_level(logging.INFO, logger=qgl.log.name):
            queries, _ = qgl.generate_queries_local(
                [_make_record(ci_id="r1")],
                cfg,
                generate_fn=gen,
                batch_size=1,
                max_retries=1,
                log_every_n=2,
            )
        progress_lines = [
            r for r in caplog.records if r.message.startswith("progress: batches=")
        ]
        assert len(progress_lines) == 3
        # Verify the milestones are in order: 2/6, 4/6, 6/6.
        msgs = [r.message for r in progress_lines]
        assert "batches=2/6" in msgs[0]
        assert "batches=4/6" in msgs[1]
        assert "batches=6/6" in msgs[2]
        # Sanity: the first line carries the right vocabulary so an
        # operator scrolling logs can find ETA at a glance.
        assert "elapsed=" in msgs[0]
        assert "avg_batch=" in msgs[0]
        assert "eta=" in msgs[0]
        assert "queries_kept=" in msgs[0]
        assert len(queries) == 6

    def test_log_every_n_zero_disables_progress(self, cfg, caplog):
        completions = ['{"query": "Q", "references": ["alpha"]}']

        def gen(batch):
            return [completions.pop(0) for _ in batch]

        with caplog.at_level(logging.INFO, logger=qgl.log.name):
            qgl.generate_queries_local(
                [_make_record(ci_id="r1", ft="alpha alpha alpha")],
                cfg,
                generate_fn=gen,
                batch_size=64,  # one batch
                max_retries=1,
                log_every_n=0,
            )
        assert not any(
            r.message.startswith("progress: batches=") for r in caplog.records
        )

    def test_log_includes_total_batches_in_planned_line(self, cfg, caplog):
        """The 'planned:' line now includes the batch count for the operator."""
        def gen(batch):
            return ['{"query": "Q", "references": ["alpha"]}' for _ in batch]

        with caplog.at_level(logging.INFO, logger=qgl.log.name):
            qgl.generate_queries_local(
                [_make_record(ci_id="r1")],
                cfg,
                generate_fn=gen,
                batch_size=4,
                max_retries=1,
            )
        planned = [r.message for r in caplog.records if "planned:" in r.message]
        assert planned, "planned: line missing from logs"
        # 6 jobs at batch=4 → ceil(6/4) = 2 batches.
        assert "batches=2" in planned[0]


# ---------------------------------------------------------------------------
# Length-sorted batching — packs similar-length prompts per batch
# ---------------------------------------------------------------------------


class TestLengthSortedBatching:
    def test_long_prompts_land_in_earlier_batches(self, cfg):
        """Longest-doc records dispatch first so each batch packs similar lengths."""
        # Three records of clearly different lengths. _plan_jobs emits
        # them in record-insertion order; the run loop sorts by
        # descending doc length before batching.
        short = qg.CorpusRecord(ci_id="short", lg="fr", ft="alpha alpha alpha")
        medium = qg.CorpusRecord(
            ci_id="medium", lg="fr", ft="bravo " * 100 + "bravo"
        )
        long = qg.CorpusRecord(
            ci_id="long", lg="fr", ft="charlie " * 1000 + "charlie"
        )

        ci_ids_seen_per_batch: list[list[str]] = []

        def gen(batch):
            # Recover ci_ids from the batched messages so the test
            # asserts on dispatch ordering, not on _plan_jobs ordering.
            ids = []
            for messages in batch:
                user = messages[1]["content"]
                # _make_record builds 'FOCUS REGION (...)\n---\n{bucket_text}\n---\n\nFULL ARTICLE...'
                # easier path: search for the unique tokens.
                if "alpha" in user:
                    ids.append("short")
                elif "bravo" in user:
                    ids.append("medium")
                elif "charlie" in user:
                    ids.append("long")
                else:
                    ids.append("?")
            ci_ids_seen_per_batch.append(ids)
            return ['{"query": "Q", "references": ["zzz"]}' for _ in batch]

        # Insertion order is short/medium/long; length order is long/medium/short.
        records = [short, medium, long]
        qgl.generate_queries_local(
            records, cfg, generate_fn=gen, batch_size=2, max_retries=1
        )
        # 18 jobs (3 records × 3 buckets × 2 query_types) at batch=2 → 9 batches.
        # First batch should be all "long" (longest doc) — same record means
        # adjacent jobs of identical length pack together cleanly.
        assert ci_ids_seen_per_batch[0] == ["long", "long"]
        # Last batch should be "short" (shortest doc).
        assert ci_ids_seen_per_batch[-1] == ["short", "short"]

    def test_empty_corpus(self, cfg):
        def gen(batch):
            raise AssertionError("should not be called")

        queries, stats = qgl.generate_queries_local(
            [], cfg, generate_fn=gen, batch_size=1, max_retries=1
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
            "--batch-size", "8",
        ])
        gen, knobs = qgl.config_from_args(args, study_cfg)
        assert gen.model == "some-other-model"
        assert gen.temperature == 0.0
        assert knobs.dtype == "fp16"
        assert knobs.attention == "flash_attention_2"
        assert knobs.max_retries == 7
        assert knobs.batch_size == 8

    def test_default_batch_size_is_one(self, study_cfg):
        """Default --batch-size = 1 keeps single-stream behaviour for laptop runs."""
        args = qgl.build_parser().parse_args(["--config", "x"])
        _, knobs = qgl.config_from_args(args, study_cfg)
        assert knobs.batch_size == qgl.DEFAULT_BATCH_SIZE == 1

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
