"""YAML-driven study configuration for the chunking-eval research.

A *study* is one coherent set of hyperparameters that answers ONE
research question. The four research CLIs (``corpus_select``,
``corpus_fetch``, ``embed_sweep``, ``query_generate``) read a single
YAML per run instead of piling per-module ``DEFAULT_*`` constants on
top of Make variables on top of CLI flags — the pattern that allowed
the chunk-grid-vs-corpus mismatch step 5 (this module) exists to
prevent.

Layout::

    configs/research/
      base.yaml              # cross-study invariants (model pin, providers, ...)
      study-v1.yaml          # frozen snapshot of pre-refactor defaults
      study-A-fit.yaml       # docs <= 8192 tokens
      study-B-overflow.yaml  # docs > 16384 tokens

Each study YAML may declare ``extends: base.yaml`` to pull in shared
defaults; the loader deep-merges the overlay onto the base before
validating. Single-level inheritance only (chained ``extends`` is
explicitly rejected — keeps reasoning local).

Path templating: ``paths.local_root`` and ``paths.s3_root`` MUST
contain a literal ``{study}`` placeholder so two studies cannot
accidentally share an S3 key or a local artefact. Filenames under
each root are convention-driven (``corpus.jsonl.bz2``,
``queries.jsonl.bz2``, ``<scenario_id>.jsonl.bz2``) — the per-CLI
S3-path flags are gone, so studying one experiment is "one prefix in,
one prefix out".

Validation: required fields (``study.name``, ``corpus.min_tokens``,
``scenarios.chunk_sizes``) have no defaults, so a YAML that forgets
them errors at load time. Cross-field invariants (``max_tokens >
min_tokens``, every language has a ``providers`` and
``chars_per_token`` entry, ``year_min <= year_max``) are enforced via
Pydantic model validators.

Design rationale, rejected alternatives, and the migration plan that
turns the four research CLIs into config-driven entry points lives in
``.progress/study-config/notes.md``.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Filenames are convention; the same names apply to the local mirror
# under ``paths.local_root`` and to the S3 keys under ``paths.s3_root``.
# Listing the whole study under one S3 prefix means ``aws s3 ls`` shows
# the full matrix at a glance — corpus, queries, and ``S0..SN`` shards.
MANIFEST_FILENAME: str = "manifest.jsonl"
CORPUS_FILENAME: str = "corpus.jsonl.bz2"
QUERIES_FILENAME: str = "queries.jsonl.bz2"
QUERIES_EMBEDDED_FILENAME: str = "queries-embedded.jsonl.bz2"


def scenario_filename(scenario_id: str) -> str:
    """Per-scenario shard filename, e.g. ``S0.jsonl.bz2``."""
    return f"{scenario_id}.jsonl.bz2"


class _Frozen(BaseModel):
    """Common config base: immutable, reject unknown keys.

    ``extra="forbid"`` is the load-time guardrail against YAML typos —
    ``langauges:`` would silently no-op without it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class StudyMeta(_Frozen):
    """``study:`` block — names the study and is the substitution
    token for every ``{study}`` placeholder in ``paths``."""

    name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description=(
            "Slug used in S3 keys, local paths, and runai job names. "
            "Restricted to [A-Za-z0-9_-] so it survives every downstream "
            "naming surface untouched."
        ),
    )


class S3Config(_Frozen):
    bucket: str = "140-processed-data-sandbox"
    rebuilt_bucket: str = "122-rebuilt-final"


class PathsConfig(_Frozen):
    """Two roots for every study artefact: one local mirror, one S3 prefix.

    Filenames under each root are convention-driven (see module-level
    ``CORPUS_FILENAME``, ``QUERIES_FILENAME``, ``MANIFEST_FILENAME``,
    and :func:`scenario_filename`); the schema only owns the root.
    Both templates MUST contain ``{study}`` so two studies cannot
    collide on either side.
    """

    local_root: str
    s3_root: str

    @field_validator("*")
    @classmethod
    def _must_template_study(cls, v: str) -> str:
        if "{study}" not in v:
            raise ValueError(
                f"path template missing '{{study}}' placeholder: {v!r}"
            )
        return v


class CorpusConfig(_Frozen):
    input_path: str
    languages: tuple[str, ...]
    ocrqa_min: float = Field(..., ge=0.0, le=1.0)
    year_min: int
    year_max: int
    # Per-language provider filter. ``None`` (YAML ``null`` or unset value
    # like ``fr:``) and an empty tuple both mean "no filter — accept all
    # providers"; a non-empty tuple is the curated allow-list.
    providers: dict[str, tuple[str, ...] | None]
    chars_per_token: dict[str, float]
    n_per_lg: int = Field(..., gt=0)
    seed: int = 42
    min_tokens: int = Field(..., gt=0)
    max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _coherent(self) -> "CorpusConfig":
        if self.max_tokens is not None and self.max_tokens <= self.min_tokens:
            raise ValueError(
                f"corpus.max_tokens ({self.max_tokens}) must be > min_tokens ({self.min_tokens})"
            )
        if self.year_min > self.year_max:
            raise ValueError(
                f"corpus.year_min ({self.year_min}) must be <= year_max ({self.year_max})"
            )
        for lg in self.languages:
            if lg not in self.providers:
                raise ValueError(f"corpus.providers missing entry for language {lg!r}")
            if lg not in self.chars_per_token:
                raise ValueError(
                    f"corpus.chars_per_token missing entry for language {lg!r}"
                )
        return self


class EmbedConfig(_Frozen):
    model_name: str
    model_revision: str
    precision: Literal["bf16", "fp32"] = "bf16"
    attention: Literal["xformers", "eager"] = "xformers"
    unpad_inputs: bool = True
    min_char_length: int = Field(default=0, ge=0)


class ScenariosConfig(_Frozen):
    """Inputs to ``scenario_builder.build_scenarios``.

    The chunker family list and chunk-size grid are the cartesian
    product expanded by the builder; ``truncate_baseline`` adds an
    extra scenario at id ``S0`` with no chunker. Chunker and aggregator
    names are NOT validated against their respective registries here —
    that responsibility lives in the builder so this module stays free
    of imports from ``impresso_text_embedder.chunking`` and
    ``impresso_text_embedder.aggregation``.

    Aggregation can be either a single value (``aggregator: mean``) or
    a list (``aggregators: [mean, max, first-chunk, length-weighted]``)
    that fans the cartesian into a third dimension. Setting both is a
    schema error — pick one. The singleton path stays bit-identical to
    the pre-list behaviour so existing studies (A-fit, B-overflow,
    v1) keep their scenario IDs and ``config_sha`` unchanged.
    """

    truncate_baseline: bool = True
    chunkers: tuple[str, ...]
    chunk_sizes: tuple[int, ...]
    aggregator: str = "mean"
    aggregators: tuple[str, ...] | None = None

    @field_validator("chunk_sizes")
    @classmethod
    def _positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("scenarios.chunk_sizes must be non-empty")
        if any(x <= 0 for x in v):
            raise ValueError(
                f"scenarios.chunk_sizes must all be > 0, got {list(v)}"
            )
        return v

    @field_validator("chunkers")
    @classmethod
    def _non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("scenarios.chunkers must be non-empty")
        return v

    @model_validator(mode="after")
    def _aggregator_xor_aggregators(self) -> "ScenariosConfig":
        # ``aggregator`` defaults to "mean", so we can't detect "set" by
        # presence; the rule is "if aggregators is given, the singular
        # MUST be left at its default". Studies that want a single
        # non-default aggregator still use ``aggregator: <name>``.
        if self.aggregators is not None:
            if not self.aggregators:
                raise ValueError("scenarios.aggregators must be non-empty when set")
            if self.aggregator != "mean":
                raise ValueError(
                    "scenarios.aggregator and scenarios.aggregators are mutually "
                    "exclusive — drop the singular form when listing multiple "
                    f"aggregators (got aggregator={self.aggregator!r}, "
                    f"aggregators={list(self.aggregators)})"
                )
        return self

    def effective_aggregators(self) -> tuple[str, ...]:
        """The aggregator list the builder will iterate over.

        ``(self.aggregator,)`` when only the singular form is set;
        ``self.aggregators`` when the plural form is set.
        """
        return self.aggregators if self.aggregators is not None else (self.aggregator,)


class QueryGenerationConfig(_Frozen):
    """Inputs to :mod:`research.query_generate`.

    ``position_buckets`` is a list of label strings; the runtime splits
    each doc's char range into ``len(position_buckets)`` contiguous
    buckets and tags each query with the matching label. Three
    buckets ("head"/"mid"/"tail") is the default; quintile or finer
    grids are a YAML edit, not a code change. Studies whose docs are
    longer (e.g. study-B-overflow at >=16k tokens) typically want
    more buckets to keep the per-bucket char span comparable across
    studies.

    ``queries_per_bucket`` is the multiplicity per (doc, bucket,
    query_type) cell. v1 ships with 1 (six queries per doc); raise
    to ``N`` for N× LLM cost and N× statistical power without
    changing the eval analysis path. ``query_id`` always carries a
    ``__NN`` sample-index suffix so consumers don't have to switch
    parsers based on the multiplicity.

    ``query_types`` is intentionally NOT a config knob — each type is
    glued to a system-prompt branch in
    :func:`research.query_generate.build_system_prompt`. Adding a
    type is a code change, not a YAML edit; surfacing it in YAML
    would lie about that.
    """

    endpoint: str
    model: str
    max_parallel: int = Field(..., gt=0)
    temperature: float = Field(..., ge=0.0)
    max_output_tokens: int = Field(..., gt=0)
    request_timeout_s: float = Field(..., gt=0.0)
    retry_attempts: int = Field(..., gt=0)
    position_buckets: tuple[str, ...]
    queries_per_bucket: int = Field(default=1, gt=0)

    @field_validator("position_buckets")
    @classmethod
    def _buckets_unique_and_non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("query_generation.position_buckets must be non-empty")
        if any(not isinstance(s, str) or not s.strip() for s in v):
            raise ValueError(
                "query_generation.position_buckets entries must be non-empty strings"
            )
        if len(set(v)) != len(v):
            raise ValueError(
                f"query_generation.position_buckets must be unique, got {list(v)}"
            )
        return v


class StudyConfig(_Frozen):
    """Top-level study config. Build with :func:`load_study_config`."""

    study: StudyMeta
    s3: S3Config = Field(default_factory=S3Config)
    paths: PathsConfig
    corpus: CorpusConfig
    embed: EmbedConfig
    scenarios: ScenariosConfig
    query_generation: QueryGenerationConfig

    def study_local_root(self) -> pathlib.Path:
        """``paths.local_root`` with ``{study}`` substituted."""
        return pathlib.Path(self.paths.local_root.format(study=self.study.name))

    def study_s3_root(self) -> str:
        """``paths.s3_root`` with ``{study}`` substituted, no trailing slash."""
        return self.paths.s3_root.format(study=self.study.name).rstrip("/")

    def local_path(self, filename: str) -> pathlib.Path:
        """Local-mirror path for one artefact under ``study_local_root()``."""
        return self.study_local_root() / filename

    def s3_key(self, filename: str) -> str:
        """S3 key for one artefact under ``study_s3_root()``."""
        return f"{self.study_s3_root()}/{filename}"

    @property
    def config_sha(self) -> str:
        """Stable 12-char fingerprint of the merged config.

        Embedded in output records alongside ``study.name`` so two runs
        of the same study with different YAML edits can be told apart
        even after artefacts land in S3. SHA256 over a sorted-keys JSON
        dump — order-independent, deterministic across Python versions.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def n_scenarios(self) -> int:
        """Total scenarios that ``scenario_builder`` will expand."""
        s = self.scenarios
        return (1 if s.truncate_baseline else 0) + (
            len(s.chunkers) * len(s.chunk_sizes) * len(s.effective_aggregators())
        )

    def summary(self) -> str:
        """Human-readable multi-line summary of what this study tests.

        Use directly (``print(cfg.summary())``) or rely on the Jupyter
        Markdown variant via ``_repr_markdown_``.
        """
        c, e, s, q = self.corpus, self.embed, self.scenarios, self.query_generation
        max_tokens = f"-{c.max_tokens}" if c.max_tokens is not None else "+"
        providers = ", ".join(
            f"{lg}={'/'.join(p) if p else 'all'}" for lg, p in c.providers.items()
        )
        cpt = ", ".join(f"{lg}={v:.2f}" for lg, v in c.chars_per_token.items())
        lines = [
            f"Study: {self.study.name}  (config_sha={self.config_sha})",
            "",
            "Corpus",
            f"  languages         : {', '.join(c.languages)}",
            f"  providers         : {providers}",
            f"  years             : {c.year_min}-{c.year_max}",
            f"  ocrqa_min         : {c.ocrqa_min}",
            f"  doc tokens        : {c.min_tokens}{max_tokens}",
            f"  docs per language : {c.n_per_lg}  (seed={c.seed})",
            f"  chars/token       : {cpt}",
            "",
            "Embed",
            f"  model     : {e.model_name}@{e.model_revision}",
            f"  precision : {e.precision}   attention: {e.attention}   unpad: {e.unpad_inputs}",
            f"  min_char_length : {e.min_char_length}",
            "",
            f"Scenarios ({self.n_scenarios()} total)",
            f"  truncate_baseline : {s.truncate_baseline}",
            f"  chunkers          : {', '.join(s.chunkers)}",
            f"  chunk_sizes       : {', '.join(str(x) for x in s.chunk_sizes)}",
            f"  aggregators       : {', '.join(s.effective_aggregators())}",
            "",
            "Query generation",
            f"  model              : {q.model}  @ {q.endpoint}",
            f"  temperature        : {q.temperature}   max_output_tokens: {q.max_output_tokens}",
            f"  max_parallel       : {q.max_parallel}",
            f"  position_buckets   : {', '.join(q.position_buckets)}",
            f"  queries_per_bucket : {q.queries_per_bucket}",
            "",
            "Paths",
            f"  local : {self.study_local_root()}",
            f"  s3    : s3://{self.s3.bucket}/{self.study_s3_root()}",
        ]
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        c, e, s, q = self.corpus, self.embed, self.scenarios, self.query_generation
        max_tokens = f"–{c.max_tokens}" if c.max_tokens is not None else "+"
        providers = ", ".join(
            f"**{lg}**: {'/'.join(p) if p else '_all_'}"
            for lg, p in c.providers.items()
        )
        cpt = ", ".join(f"**{lg}**={v:.2f}" for lg, v in c.chars_per_token.items())
        chunkers = ", ".join(f"`{x}`" for x in s.chunkers)
        sizes = ", ".join(f"`{x}`" for x in s.chunk_sizes)
        aggs = ", ".join(f"`{x}`" for x in s.effective_aggregators())
        return (
            f"### Study `{self.study.name}` &nbsp;<sub>config_sha=`{self.config_sha}`</sub>\n\n"
            f"**Corpus** — {', '.join(c.languages)} · {c.year_min}–{c.year_max} · "
            f"ocrqa ≥ {c.ocrqa_min} · doc tokens {c.min_tokens}{max_tokens} · "
            f"{c.n_per_lg}/lg (seed {c.seed})\n\n"
            f"- providers: {providers}\n"
            f"- chars/token: {cpt}\n\n"
            f"**Embed** — `{e.model_name}@{e.model_revision}` · {e.precision} · "
            f"attn={e.attention} · unpad={e.unpad_inputs} · min_char_length={e.min_char_length}\n\n"
            f"**Scenarios** — {self.n_scenarios()} total · "
            f"truncate_baseline={s.truncate_baseline}\n\n"
            f"- chunkers: {chunkers}\n"
            f"- chunk_sizes: {sizes}\n"
            f"- aggregators: {aggs}\n\n"
            f"**Query generation** — `{q.model}` @ `{q.endpoint}` · "
            f"T={q.temperature} · max_parallel={q.max_parallel} · "
            f"{q.queries_per_bucket}/bucket × ({', '.join(q.position_buckets)})\n\n"
            f"**Paths** — local: `{self.study_local_root()}` · "
            f"s3: `s3://{self.s3.bucket}/{self.study_s3_root()}`\n"
        )


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``.

    Dict values are merged recursively; lists, scalars, and tuples are
    replaced outright (no append semantics — overlay wins). Mirrors
    Docker Compose ``extends`` and Hydra's defaults override.
    """
    out: dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_study_config(path: str | pathlib.Path) -> StudyConfig:
    """Read ``path``, resolve a single-level ``extends:``, validate, return.

    ``yaml`` is imported lazily so test code that builds configs
    in-memory via ``StudyConfig.model_validate(...)`` does not pay the
    parser's import cost.
    """
    import yaml

    p = pathlib.Path(path).resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{p}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )

    extends = raw.pop("extends", None)
    if extends is not None:
        if not isinstance(extends, str):
            raise ValueError(
                f"{p}: 'extends' must be a string filename, got {type(extends).__name__}"
            )
        base_path = (p.parent / extends).resolve()
        if base_path == p:
            raise ValueError(f"{p}: 'extends: {extends}' points at self")
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        if not isinstance(base_raw, dict):
            raise ValueError(
                f"{base_path}: top-level YAML must be a mapping, got "
                f"{type(base_raw).__name__}"
            )
        if "extends" in base_raw:
            raise ValueError(
                f"{p}: chained 'extends' is not supported "
                f"(base {base_path.name} also declares 'extends')"
            )
        raw = _deep_merge(base_raw, raw)

    return StudyConfig.model_validate(raw)


__all__ = [
    "CORPUS_FILENAME",
    "CorpusConfig",
    "EmbedConfig",
    "MANIFEST_FILENAME",
    "PathsConfig",
    "QUERIES_FILENAME",
    "QueryGenerationConfig",
    "S3Config",
    "ScenariosConfig",
    "StudyConfig",
    "StudyMeta",
    "load_study_config",
    "scenario_filename",
]
