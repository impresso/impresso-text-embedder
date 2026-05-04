"""Generate position-stratified synthetic queries for the chunking-eval.

Step 4 of ``research/chunking-eval``: read the corpus shard built by
:mod:`research.corpus_fetch`, ask an LLM (via EPFL RCP AIaaS, an
OpenAI-compatible endpoint) to produce ``(query, references)`` JSON
for each (doc, position-bucket, query-type) tuple, verify the
references appear verbatim inside the targeted bucket region of the
source doc, and write a single ``.jsonl.bz2`` of surviving queries
to ``<study_s3_root>/queries.jsonl.bz2``.

The LLM call uses LangChain's :class:`ChatOpenAI` with
``with_structured_output(QueryOutput, method="json_mode")`` so each
completion is parsed into a Pydantic model (no hand-rolled fence
stripping) and ``.with_retry()`` so transient HTTP / parsing
failures are retried with exponential backoff. Methodology mirrors
`Chroma's chunking-eval protocol
<https://research.trychroma.com/evaluating-chunking>`_; design
rationale, rejected alternatives, and open items in
``.progress/query-generation/notes.md``.

CLI surface is config-driven via ``--config <path>``; corpus and
output paths derive from the study YAML. ``--no-upload`` writes the
queries shard to the local mirror (``paths.local_root +
queries.jsonl.bz2``) instead of uploading.
"""

from __future__ import annotations

import argparse
import asyncio
import bz2
import dataclasses
import datetime as dt
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, Field
from tqdm import tqdm

from impresso_text_embedder import io as s3io
from impresso_text_embedder.research._io import staged_input, staged_output
from impresso_text_embedder.research.study_config import (
    CORPUS_FILENAME,
    QUERIES_FILENAME,
    load_study_config,
)

log = logging.getLogger(__name__)


DEFAULT_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_ENDPOINT: str = "https://inference.rcp.epfl.ch/v1"
DEFAULT_MAX_PARALLEL: int = 2  # AIaaS per-key cap
DEFAULT_TEMPERATURE: float = 0.7
# 1500 tokens covers a query plus 1-3 long verbatim newspaper-quote
# references with comfortable headroom. Smaller values caused the model
# to emit truncated JSON ("Unterminated string …") on long-quote
# generations during the first --limit 10 smoke run.
DEFAULT_MAX_OUTPUT_TOKENS: int = 1500
DEFAULT_REQUEST_TIMEOUT_S: float = 120.0
DEFAULT_RETRY_ATTEMPTS: int = 5

QueryType = Literal["question", "topical-phrase"]
# Bucket labels are config-driven via GenerationConfig.position_buckets.
# Three is the default; quintile or finer grids are a YAML edit. Adding
# a new query type, by contrast, requires a new prompt branch in
# build_system_prompt, so QUERY_TYPES stays a code-coupled constant.
DEFAULT_POSITION_BUCKETS: tuple[str, ...] = ("head", "mid", "tail")
DEFAULT_QUERIES_PER_BUCKET: int = 1
QUERY_TYPES: tuple[QueryType, QueryType] = ("question", "topical-phrase")

# Above this many characters, the FULL ARTICLE block is dropped from
# build_user_message and only the FOCUS REGION is sent. Rule 1 of the
# system prompt already constrains the query to the focus region, so the
# full article is soft context (only useful for the "fact specific to
# this region" check). On long-doc studies (study-B-overflow, 16–60k
# tokens / ~64–240k chars) keeping it doubled the prompt and timed out
# the EPFL endpoint at the 120 s default. ~32 k chars ≈ the embedder's
# 8190-token context at fr/de chars-per-token — the project's natural
# cutoff between "fits in one shot" and "long".
_INCLUDE_FULL_ARTICLE_CHAR_LIMIT: int = 32_000

_LG_DISPLAY: dict[str, str] = {
    "fr": "French",
    "de": "German",
    "lb": "Luxembourgish",
    "it": "Italian",
    "en": "English",
}


# In-language few-shot exemplars anchor search-bar style (compound nouns
# in DE, accent-stripped keyword strings in FR) that English-only
# instructions tend to miss. Rules stay in English so instruction
# compliance doesn't degrade; only the example flips to the target
# language. lb falls back to fr (see ``build_system_prompt``) — Qwen3
# lb-instruction-following is weaker than its lb-generation, and lb
# users frequently search in fr anyway. it/en have no exemplar today.
_EXEMPLARS: dict[str, dict[QueryType, str]] = {
    "fr": {
        "question": (
            "Example output (do not copy; learn the style — note the "
            "interrogative form ending with '?', and the capitalised "
            "place name 'Berne'):\n"
            '{"query": "Comment Berne a-t-elle révisé sa constitution '
            'cantonale en 1846 ?", '
            '"references": ["la constitution cantonale fut adoptée '
            'par 34,079 citoyens contre 1,257 rejetants"]}'
        ),
        "topical-phrase": (
            "Example output (do not copy; learn the style — short "
            "keyword phrase, lowercase common nouns, capitalised "
            "proper noun 'Berne'):\n"
            '{"query": "révision constitution cantonale Berne 1846", '
            '"references": ["constitution cantonale révisée"]}'
        ),
    },
    "de": {
        "question": (
            "Example output (do not copy; learn the style — note the "
            "interrogative form ending with '?', and the capitalised "
            "place name 'Bern'; German common nouns are also "
            "capitalised):\n"
            '{"query": "Wie hat Bern 1846 seine Kantonsverfassung '
            'revidiert?", '
            '"references": ["die Kantonsverfassung wurde 1846 von '
            '34 079 Bürgern angenommen"]}'
        ),
        "topical-phrase": (
            "Example output (do not copy; learn the style — German "
            "nouns capitalised per German convention):\n"
            '{"query": "Revision Kantonsverfassung Bern 1846", '
            '"references": ["neue Kantonsverfassung"]}'
        ),
    },
}


# ---------------------------------------------------------------------------
# Schemas — Pydantic for the LLM output, dataclasses for everything else
# ---------------------------------------------------------------------------


class QueryOutput(BaseModel):
    """Per-call LLM output schema.

    Mirrors the JSON contract documented in the system prompt: one
    natural-language query plus 1-3 verbatim substrings of the source
    that support its answer. Used by
    :meth:`langchain_openai.ChatOpenAI.with_structured_output` so a
    malformed completion raises a parse error before reaching
    verification, instead of silently producing a Query with garbage
    references.
    """

    query: str = Field(..., description="The query text in the source language.")
    references: list[str] = Field(
        default_factory=list,
        description="Verbatim substrings of the source article supporting the answer.",
    )


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    """Inputs for :func:`generate_queries`."""

    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    api_key: str = ""
    max_parallel: int = DEFAULT_MAX_PARALLEL
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    limit: int | None = None
    study_name: str | None = None
    study_config_sha: str | None = None
    position_buckets: tuple[str, ...] = DEFAULT_POSITION_BUCKETS
    queries_per_bucket: int = DEFAULT_QUERIES_PER_BUCKET


@dataclasses.dataclass(frozen=True)
class CorpusRecord:
    ci_id: str
    lg: str
    ft: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CorpusRecord:
        return cls(
            ci_id=str(raw["ci_id"]),
            lg=str(raw.get("lg") or ""),
            ft=str(raw.get("ft") or ""),
        )


@dataclasses.dataclass(frozen=True)
class Reference:
    text: str
    char_start: int
    char_end: int

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Query:
    query_id: str
    ci_id: str
    lg: str
    query_text: str
    query_type: QueryType
    references: tuple[Reference, ...]
    position_bucket: str
    position_chars: tuple[int, int]
    gen_model: str
    gen_endpoint: str
    ts: str
    study_name: str | None = None
    study_config_sha: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query_id": self.query_id,
            "ci_id": self.ci_id,
            "lg": self.lg,
            "query_text": self.query_text,
            "query_type": self.query_type,
            "references": [r.to_jsonable() for r in self.references],
            "position_bucket": self.position_bucket,
            "position_chars": list(self.position_chars),
            "gen_model": self.gen_model,
            "gen_endpoint": self.gen_endpoint,
            "ts": self.ts,
        }
        if self.study_name is not None:
            out["study_name"] = self.study_name
        if self.study_config_sha is not None:
            out["study_config_sha"] = self.study_config_sha
        return out


@dataclasses.dataclass(frozen=True)
class Job:
    record: CorpusRecord
    bucket_idx: int
    query_type: QueryType
    bucket_range: tuple[int, int]
    bucket_text: str
    bucket_label: str = ""
    sample_idx: int = 0


@dataclasses.dataclass(frozen=True)
class JobResult:
    """Outcome of one :func:`generate_one` call, folded into stats by the reducer."""

    query: Query | None
    error_kind: Literal["ok", "api", "no_refs"] = "ok"
    refs_not_found: int = 0
    refs_out_of_bucket: int = 0


@dataclasses.dataclass
class GenerationStats:
    corpus_records: int = 0
    attempts: int = 0
    api_errors: int = 0
    no_refs_returned: int = 0
    refs_out_of_bucket: int = 0
    refs_not_found: int = 0
    queries_kept: int = 0
    by_bucket: dict[str, int] = dataclasses.field(default_factory=dict)
    by_lg: dict[str, int] = dataclasses.field(default_factory=dict)
    by_query_type: dict[str, int] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def bucket_ranges(total_chars: int, n: int = 3) -> tuple[tuple[int, int], ...]:
    """Return ``n`` contiguous (lo, hi) char ranges spanning the doc.

    The last bucket extends to ``total_chars`` so any remainder from
    integer division lands in the tail bucket; earlier buckets are
    exact ``total_chars // n`` wide. Empty docs (``total_chars <= 0``)
    return ``n`` ``(0, 0)`` slots, which :func:`_plan_jobs` then
    filters out so the LLM never sees an empty focus region.

    ``n`` defaults to 3 to preserve the original head/mid/tail split
    when called without a bucket count; callers driving from
    :class:`GenerationConfig.position_buckets` should pass
    ``len(cfg.position_buckets)``.
    """
    if n <= 0:
        raise ValueError(f"bucket_ranges: n must be > 0, got {n}")
    if total_chars <= 0:
        return tuple((0, 0) for _ in range(n))
    step = total_chars // n
    out: list[tuple[int, int]] = []
    for i in range(n):
        lo = i * step
        hi = total_chars if i == n - 1 else (i + 1) * step
        out.append((lo, hi))
    return tuple(out)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_BASE_RULES = (
    "Rules:\n"
    "1. The query must be answerable from the FOCUS REGION below; do not\n"
    "   draw on content from outside the focus region. Where the same\n"
    "   fact also appears outside the focus region, prefer a fact that\n"
    "   is specific to the focus region.\n"
    "2. Provide 1 to 3 short references — verbatim substrings of the\n"
    "   article that support the answer. Aim for non-overlapping,\n"
    "   non-adjacent spans, each at most ~180 characters (one or two\n"
    "   short sentences). Each reference must appear in the article\n"
    "   EXACTLY as written: same casing, same punctuation, same archaic\n"
    "   spelling, same OCR artefacts. Do not modernise, paraphrase,\n"
    "   fix typos, or skip across hyphenation.\n"
    "3. The query must be in {language}.\n"
    "4. Reply with EXACTLY this JSON shape and no other keys:\n"
    '   {{"query": "<the {language} query>", '
    '"references": ["<verbatim span 1>", "<verbatim span 2>"]}}'
)


def build_system_prompt(
    query_type: QueryType, lg: str, sample_idx: int = 0
) -> str:
    language = _LG_DISPLAY.get(lg, "English")
    if query_type == "question":
        intro = (
            "You write retrieval-evaluation queries for a historical "
            "newspaper archive. Given a long article and a focus region "
            "inside it, write ONE realistic question (at most 20 "
            "words) a researcher would type to find this article "
            "WITHOUT having read it yet. The query MUST be a real "
            "question: start with an interrogative word (\"comment\", "
            "\"qu'est-ce que\", \"quel\", \"où\", \"quand\", \"wie\", "
            "\"was\", \"warum\", \"wer\", \"how\", \"what\", etc.) "
            "and end with a question mark. Do NOT emit a keyword "
            "string — that is the topical-phrase type, not this one. "
            "Do not refer to the article (\"selon l'article\", "
            "\"according to the article\", \"laut dem Artikel\"). Do "
            "not name a person, place, or date unless they are widely "
            "known at public-history level — never a name introduced "
            "only inside the article. Where natural, paraphrase the "
            "topic instead of copying multi-word phrases verbatim "
            "from the focus region."
        )
    elif query_type == "topical-phrase":
        intro = (
            "You write retrieval-evaluation queries for a historical "
            "newspaper archive. Given a long article and a focus "
            "region inside it, write ONE topical search phrase (3 to "
            "6 keywords, no question mark, no leading function words "
            "like \"le \", \"la \", \"pas de \", \"des \", \"the \"). "
            "Keywords only — no full clauses or sentences. Use "
            "natural capitalisation for the target language: "
            "capitalise place names (\"Berne\", \"Afrique\", "
            "\"Italie\"), person names (\"Strickland\"), and country "
            "names; lowercase common nouns in French and Italian; in "
            "German, common nouns are also capitalised per German "
            "convention. Avoid named individuals unless widely known "
            "at public-history level. Where natural, paraphrase the "
            "topic instead of copying multi-word phrases verbatim "
            "from the focus region."
        )
    else:
        raise ValueError(f"unknown query_type: {query_type!r}")

    # lb routes through the FR exemplar (see _EXEMPLARS comment); other
    # languages either find their own exemplar or fall through with rules
    # only — same behaviour as before plus the rule tightening.
    exemplar_lg = "fr" if lg == "lb" else lg
    exemplar = _EXEMPLARS.get(exemplar_lg, {}).get(query_type)
    parts = [intro, _BASE_RULES.format(language=language)]
    if exemplar is not None:
        parts.append(exemplar)
    return "\n\n".join(parts)


def build_user_message(
    record: CorpusRecord,
    bucket_label: str,
    bucket_text: str,
    sample_idx: int = 0,
) -> str:
    diversity = ""
    if sample_idx > 0:
        diversity = (
            f"\n\nThis is sample {sample_idx + 1}. Choose a different "
            f"angle and different supporting facts than would be the "
            f"obvious first choice."
        )
    blocks = [
        f"FOCUS REGION ({bucket_label} of the article):\n"
        f"---\n{bucket_text}\n---"
    ]
    if len(record.ft) <= _INCLUDE_FULL_ARTICLE_CHAR_LIMIT:
        blocks.append(f"FULL ARTICLE:\n---\n{record.ft}\n---")
    return (
        "\n\n".join(blocks)
        + diversity
        + "\n\n"
        + f"Pick a SPECIFIC fact from the {bucket_label} focus region "
        + "(a number, a named development, a particular event) — not "
        + "the article's overall theme — so the query is distinctive "
        + "to this region and would not equally fit a different "
        + "region of the same article.\n\n"
        + "Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# LLM client — LangChain ChatOpenAI + structured output + retry
# ---------------------------------------------------------------------------


def make_llm(cfg: GenerationConfig):
    """Build a LangChain runnable that returns :class:`QueryOutput` per call.

    Combines three LangChain primitives:

    1. :class:`ChatOpenAI` — the OpenAI-compatible chat client; the
       ``base_url`` knob makes it work against AIaaS / vLLM / any
       endpoint that speaks the OpenAI chat protocol.
    2. ``with_structured_output(QueryOutput, method="json_mode")`` —
       sets ``response_format={"type": "json_object"}`` on the API
       call AND parses the raw completion into a :class:`QueryOutput`
       Pydantic instance, raising on malformed JSON or schema
       violations. We pick ``json_mode`` over ``function_calling``
       because vLLM-hosted Qwen3 supports the former universally.
    3. ``with_retry(stop_after_attempt=N, wait_exponential_jitter=True)``
       — retries on any exception (HTTP transient, parse error)
       with exponential backoff + jitter before bubbling up.

    Imports are local to keep module load lean for tests that don't
    exercise the live LLM path.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=cfg.endpoint,
        api_key=cfg.api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
        timeout=cfg.request_timeout_s,
    )
    structured = llm.with_structured_output(QueryOutput, method="json_mode")
    return structured.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=cfg.retry_attempts,
    )


# ---------------------------------------------------------------------------
# Reference verification (verbatim, in-bucket)
# ---------------------------------------------------------------------------


def verify_references(
    ft: str,
    refs: Sequence[str],
    bucket_range: tuple[int, int],
) -> tuple[list[Reference], int, int]:
    """Resolve each reference to a verbatim in-bucket span.

    Returns ``(verified, refs_not_found, refs_out_of_bucket)``. We pick
    the first occurrence whose start offset falls inside the bucket;
    earlier out-of-bucket matches are tracked separately so stats can
    distinguish a missing-from-source reference from a wrong-bucket one.
    """
    lo, hi = bucket_range
    verified: list[Reference] = []
    not_found = 0
    out_of_bucket = 0
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            not_found += 1
            continue
        pos = 0
        first_pos = -1
        in_bucket_pos = -1
        while True:
            idx = ft.find(ref, pos)
            if idx < 0:
                break
            if first_pos < 0:
                first_pos = idx
            if lo <= idx < hi:
                in_bucket_pos = idx
                break
            pos = idx + 1
        if in_bucket_pos >= 0:
            verified.append(
                Reference(
                    text=ref,
                    char_start=in_bucket_pos,
                    char_end=in_bucket_pos + len(ref),
                )
            )
        elif first_pos >= 0:
            out_of_bucket += 1
        else:
            not_found += 1
    return verified, not_found, out_of_bucket


# ---------------------------------------------------------------------------
# Per-job generation
# ---------------------------------------------------------------------------


def _plan_jobs(records: Sequence[CorpusRecord], cfg: GenerationConfig) -> list[Job]:
    """Enumerate one :class:`Job` per (record, bucket, query_type, sample).

    Bucket count and labels come from ``cfg.position_buckets``;
    multiplicity per (record, bucket, query_type) cell comes from
    ``cfg.queries_per_bucket``. With the v1 defaults
    (3 buckets × 2 query_types × 1 sample) that's 6 jobs per record.
    Empty buckets (very short docs) are filtered here so the per-job
    fast path can assume ``bucket_text`` is non-empty.
    """
    n_buckets = len(cfg.position_buckets)
    jobs: list[Job] = []
    for record in records:
        for bucket_idx, (lo, hi) in enumerate(bucket_ranges(len(record.ft), n_buckets)):
            if hi <= lo:
                continue
            bucket_text = record.ft[lo:hi]
            bucket_label = cfg.position_buckets[bucket_idx]
            for query_type in QUERY_TYPES:
                for sample_idx in range(cfg.queries_per_bucket):
                    jobs.append(
                        Job(
                            record=record,
                            bucket_idx=bucket_idx,
                            query_type=query_type,
                            bucket_range=(lo, hi),
                            bucket_text=bucket_text,
                            bucket_label=bucket_label,
                            sample_idx=sample_idx,
                        )
                    )
    return jobs


async def generate_one(job: Job, cfg: GenerationConfig, llm: Any) -> JobResult:
    """Generate one query for ``job``; return :class:`JobResult` for the reducer.

    ``llm`` is a LangChain runnable produced by :func:`make_llm` (or a
    test stub with the same ``ainvoke`` interface). The runnable
    handles JSON parsing and retries internally; here we only deal
    with verifying that the returned references actually appear
    verbatim inside the source bucket.
    """
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                job.query_type, job.record.lg, job.sample_idx
            ),
        },
        {
            "role": "user",
            "content": build_user_message(
                job.record, job.bucket_label, job.bucket_text, job.sample_idx
            ),
        },
    ]

    try:
        parsed: QueryOutput = await llm.ainvoke(messages)
    except Exception as exc:  # noqa: BLE001 — surfacing per-call failure
        log.warning(
            "llm error for ci_id=%s bucket=%s type=%s: %s",
            job.record.ci_id,
            job.bucket_label,
            job.query_type,
            exc,
        )
        return JobResult(query=None, error_kind="api")

    verified, not_found, oob = verify_references(
        job.record.ft, parsed.references, job.bucket_range
    )
    if not verified:
        return JobResult(
            query=None,
            error_kind="no_refs",
            refs_not_found=not_found,
            refs_out_of_bucket=oob,
        )

    # query_id always carries a __NN sample-index suffix so consumers
    # don't have to switch parsers based on whether the run used
    # queries_per_bucket > 1. The leading 0 keeps lexicographic sort
    # order matching insertion order up to N=99.
    query = Query(
        query_id=(
            f"{job.record.ci_id}__{job.bucket_label}__{job.query_type}"
            f"__{job.sample_idx:02d}"
        ),
        ci_id=job.record.ci_id,
        lg=job.record.lg,
        query_text=parsed.query.strip(),
        query_type=job.query_type,
        references=tuple(verified),
        position_bucket=job.bucket_label,
        position_chars=job.bucket_range,
        gen_model=cfg.model,
        gen_endpoint=cfg.endpoint,
        ts=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        study_name=cfg.study_name,
        study_config_sha=cfg.study_config_sha,
    )
    return JobResult(query=query, refs_not_found=not_found, refs_out_of_bucket=oob)


# ---------------------------------------------------------------------------
# Async run loop
# ---------------------------------------------------------------------------


async def generate_queries(
    records: Sequence[CorpusRecord],
    cfg: GenerationConfig,
    llm: Any,
) -> tuple[list[Query], GenerationStats]:
    """Drive generation across all (doc, bucket, query_type) jobs concurrently.

    A single ``asyncio.Semaphore(cfg.max_parallel)`` enforces the
    AIaaS per-key cap. ``asyncio.gather`` preserves input order so
    the output queries land in deterministic
    ``(ci_id, bucket_idx, query_type)`` order.
    """
    stats = GenerationStats(corpus_records=len(records))
    jobs = _plan_jobs(records, cfg)
    if not jobs:
        return [], stats

    log.info(
        "planned: corpus_records=%d jobs=%d buckets=%s queries_per_bucket=%d "
        "max_parallel=%d model=%s",
        stats.corpus_records,
        len(jobs),
        list(cfg.position_buckets),
        cfg.queries_per_bucket,
        cfg.max_parallel,
        cfg.model,
    )

    sem = asyncio.Semaphore(max(1, cfg.max_parallel))
    pbar = tqdm(total=len(jobs), file=sys.stderr, disable=None, desc="generate", unit="q")

    async def run(job: Job) -> JobResult:
        async with sem:
            try:
                return await generate_one(job, cfg, llm)
            finally:
                pbar.update(1)

    try:
        results = await asyncio.gather(*(run(job) for job in jobs))
    finally:
        pbar.close()

    queries: list[Query] = []
    for result in results:
        stats.attempts += 1
        stats.refs_not_found += result.refs_not_found
        stats.refs_out_of_bucket += result.refs_out_of_bucket
        if result.query is not None:
            queries.append(result.query)
            stats.queries_kept += 1
            q = result.query
            stats.by_bucket[q.position_bucket] = stats.by_bucket.get(q.position_bucket, 0) + 1
            stats.by_lg[q.lg] = stats.by_lg.get(q.lg, 0) + 1
            stats.by_query_type[q.query_type] = stats.by_query_type.get(q.query_type, 0) + 1
        elif result.error_kind == "api":
            stats.api_errors += 1
        elif result.error_kind == "no_refs":
            stats.no_refs_returned += 1
    return queries, stats


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_corpus_shard(path: Path) -> list[CorpusRecord]:
    return [
        CorpusRecord.from_dict(orjson.loads(line))
        for line in s3io.iter_jsonl_bz2_path(path)
    ]


def write_queries(queries: Iterable[Query], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wb") as fh:
        for q in queries:
            fh.write(orjson.dumps(q.to_jsonable(), option=orjson.OPT_APPEND_NEWLINE))


def _format_stats(stats: GenerationStats) -> str:
    parts = [
        f"corpus_records={stats.corpus_records}",
        f"attempts={stats.attempts}",
        f"queries_kept={stats.queries_kept}",
        f"api_errors={stats.api_errors}",
        f"no_refs_returned={stats.no_refs_returned}",
        f"refs_not_found={stats.refs_not_found}",
        f"refs_out_of_bucket={stats.refs_out_of_bucket}",
    ]
    for label, table in (
        ("by_bucket", stats.by_bucket),
        ("by_lg", stats.by_lg),
        ("by_query_type", stats.by_query_type),
    ):
        if table:
            parts.append(f"{label}=(" + " ".join(f"{k}={v}" for k, v in sorted(table.items())) + ")")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-research-query-generate",
        description=(
            "Generate position-stratified synthetic queries from the "
            "chunking-eval corpus shard via the EPFL RCP AIaaS endpoint."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Study YAML (e.g. configs/research/study-v1.yaml). Source of "
            "truth for corpus/output paths and query_generation.* defaults; "
            "per-flag CLI args still override the LLM knobs field-by-field."
        ),
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "skip the S3 upload; the queries shard lands at the study's "
            "local mirror (paths.local_root + queries.jsonl.bz2) so the "
            "user can find it deterministically."
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--api-key", default=None, help="override the RCP_API_KEY env var")
    p.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    p.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    p.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    p.add_argument(
        "--queries-per-bucket",
        type=int,
        default=DEFAULT_QUERIES_PER_BUCKET,
        help=(
            "multiplicity per (doc, bucket, query_type) cell; default 1. "
            "Raise to N for N x LLM cost and N x statistical power."
        ),
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    return p


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in dependencies
        return
    load_dotenv()


def config_from_args(args: argparse.Namespace, study_cfg) -> GenerationConfig:
    api_key = args.api_key or os.environ.get("RCP_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "RCP_API_KEY is not set. Add it to .env or pass --api-key. "
            "Get a key from https://portal.rcp.epfl.ch/aiaas/keys."
        )

    study_qg = study_cfg.query_generation

    def _pick(cli_val: object, default_val: object, cfg_val: object) -> object:
        # CLI matches its argparse default → take YAML; otherwise CLI wins.
        if cli_val == default_val:
            return cfg_val
        return cli_val

    model = _pick(args.model, DEFAULT_MODEL, study_qg.model)
    endpoint = _pick(args.endpoint, DEFAULT_ENDPOINT, study_qg.endpoint)
    max_parallel = int(_pick(
        args.max_parallel, DEFAULT_MAX_PARALLEL, study_qg.max_parallel
    ))
    temperature = float(_pick(
        args.temperature, DEFAULT_TEMPERATURE, study_qg.temperature
    ))
    max_output_tokens = int(_pick(
        args.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS, study_qg.max_output_tokens
    ))
    request_timeout_s = float(_pick(
        args.request_timeout_s, DEFAULT_REQUEST_TIMEOUT_S, study_qg.request_timeout_s
    ))
    retry_attempts = int(_pick(
        args.retry_attempts, DEFAULT_RETRY_ATTEMPTS, study_qg.retry_attempts
    ))
    queries_per_bucket = int(_pick(
        args.queries_per_bucket, DEFAULT_QUERIES_PER_BUCKET, study_qg.queries_per_bucket
    ))

    return GenerationConfig(
        model=str(model),
        endpoint=str(endpoint),
        api_key=api_key,
        max_parallel=max_parallel,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        request_timeout_s=request_timeout_s,
        retry_attempts=retry_attempts,
        limit=args.limit,
        study_name=study_cfg.study.name,
        study_config_sha=study_cfg.config_sha,
        position_buckets=tuple(study_qg.position_buckets),
        queries_per_bucket=queries_per_bucket,
    )


async def _run(
    cfg: GenerationConfig,
    bucket: str,
    corpus_key: str,
    out_key: str,
    out_local_mirror: Path,
    *,
    upload: bool,
) -> int:
    with staged_input(bucket, corpus_key) as corpus_path:
        records = read_corpus_shard(corpus_path)
    if cfg.limit is not None:
        records = records[: cfg.limit]
    log.info("loaded %d corpus records", len(records))

    llm = make_llm(cfg)
    queries, stats = await generate_queries(records, cfg, llm)
    log.info("generation stats: %s", _format_stats(stats))

    with staged_output(bucket, out_key, out_local_mirror, upload=upload) as out_path:
        write_queries(queries, out_path)
        log.info("queries written: %s (%d queries)", out_path, len(queries))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = args.log_level.upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # httpx logs one INFO line per request; with max_parallel LLM calls
    # they shred the tqdm bar. Keep WARNING+ unless the user asked for DEBUG.
    if level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
    _load_env()

    study_cfg = load_study_config(args.config)
    cfg = config_from_args(args, study_cfg)

    bucket = study_cfg.s3.bucket
    corpus_key = study_cfg.s3_key(CORPUS_FILENAME)
    out_key = study_cfg.s3_key(QUERIES_FILENAME)
    out_local_mirror = study_cfg.local_path(QUERIES_FILENAME)

    log.info(
        "query-generate start: study=%s model=%s endpoint=%s max_parallel=%d "
        "corpus=s3://%s/%s output=s3://%s/%s upload=%s",
        cfg.study_name or "(none)",
        cfg.model,
        cfg.endpoint,
        cfg.max_parallel,
        bucket,
        corpus_key,
        bucket,
        out_key,
        "no" if args.no_upload else "yes",
    )
    return asyncio.run(
        _run(
            cfg,
            bucket=bucket,
            corpus_key=corpus_key,
            out_key=out_key,
            out_local_mirror=out_local_mirror,
            upload=not args.no_upload,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
