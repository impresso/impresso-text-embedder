"""Corpus-manifest builder for the chunking-eval research.

Streams the langident-aggregated jsonl (one record per content item with its
language, char length, OCR-quality score, content type, and source location),
filters to long, high-OCR articles in the target languages from a curated
provider subset within a year window, and samples N records per language to
produce a deterministic manifest the downstream chunking-strategy sweeps
consume.

Design rationale, parameter choices, and rejected alternatives in
``.progress/corpus-selection/notes.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

log = logging.getLogger(__name__)

# Per-language chars-per-token estimate for the gte-multilingual-base
# (XLM-Roberta SentencePiece) tokenizer. Used to convert the user-facing
# ``--min-tokens`` threshold into a char-length cutoff applied to the ``len``
# field. Empirical refinement of these values is a deferred deliverable
# (the chars_per_token-calibration experiment), and the manifest re-emits the
# table that produced it.
DEFAULT_CHARS_PER_TOKEN: dict[str, float] = {
    "fr": 4.5,
    "de": 3.5,
    "lb": 3.5,
}

# Curated providers per target language: the topic/style-bounding mechanism
# that keeps the LLM query-generation cost manageable while preserving some
# editorial diversity. The lists below are the providers that actually carry
# long, high-OCR articles in the aggregated corpus (measured on a 1/50
# stratified scan in step 1) — NOT the headline newspaper names you would
# expect (NZZ/SWA/SUB barely surface in the article-level partition; SNL is
# the dominant multilingual carrier). Justification in
# `.progress/corpus-selection/notes.md`.
DEFAULT_PROVIDERS: dict[str, tuple[str, ...] | None] = {
    "fr": ("BNF", "LeTemps", "SNL", "FedGaz"),
    "de": ("SNL", "FedGaz", "BNL"),
    "lb": ("BNL",),
}

# lb is intentionally NOT in the default sweep: the Impresso corpus contains
# zero long (>=4000 token) Luxembourgish articles passing ocrqa>=0.9 — long
# lb articles all sit at ocrqa 0.65-0.67. Pass `--languages fr de lb` with a
# relaxed `--ocrqa-min` to opt back in for a side experiment. Empirical
# evidence and rejected alternatives in `.progress/corpus-selection/notes.md`.
DEFAULT_LANGUAGES: tuple[str, ...] = ("fr", "de")
DEFAULT_OCRQA_MIN: float = 0.9
DEFAULT_MIN_TOKENS: int = 4000
DEFAULT_N_PER_LG: int = 200
DEFAULT_YEAR_MIN: int = 1880
DEFAULT_YEAR_MAX: int = 1980
DEFAULT_SEED: int = 42

REBUILT_BUCKET: str = "122-rebuilt-final"
ARTICLE_TP: str = "article"


@dataclasses.dataclass(frozen=True)
class SelectionConfig:
    """Inputs for ``select_corpus``.

    Every dimension is overridable from the CLI; the dataclass is the
    intermediate representation so callers can construct it directly in
    tests / notebooks without going through argparse.
    """

    input_path: Path
    output_path: Path
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    n_per_lg: int = DEFAULT_N_PER_LG
    ocrqa_min: float = DEFAULT_OCRQA_MIN
    min_tokens: int = DEFAULT_MIN_TOKENS
    max_tokens: int | None = None
    chars_per_token: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CHARS_PER_TOKEN)
    )
    # Per-language provider filter. ``None`` or empty tuple = no filter
    # (accept any provider for that language). Non-empty = curated allow-list.
    providers: Mapping[str, tuple[str, ...] | None] = dataclasses.field(
        default_factory=lambda: {
            k: (tuple(v) if v else None) for k, v in DEFAULT_PROVIDERS.items()
        }
    )
    year_min: int = DEFAULT_YEAR_MIN
    year_max: int = DEFAULT_YEAR_MAX
    seed: int = DEFAULT_SEED

    def char_threshold(self, lg: str) -> int:
        """Min char-length cutoff for ``lg``, derived from ``min_tokens``."""
        return int(self.min_tokens * self.chars_per_token[lg])

    def char_ceiling(self, lg: str) -> int | None:
        """Max char-length cutoff for ``lg``, ``None`` when no upper bound.

        Mirrors :meth:`char_threshold` but for the upper edge of the
        length window. Study A (docs that fit in context) needs both
        bounds; the legacy v1 default leaves it unset.
        """
        if self.max_tokens is None:
            return None
        return int(self.max_tokens * self.chars_per_token[lg])


@dataclasses.dataclass(frozen=True)
class ManifestEntry:
    """One selected article with everything needed to fetch its rebuilt text."""

    ci_id: str
    lg: str
    year: int
    len_chars: int
    ocrqa: float
    provider: str
    alias: str
    rebuilt_bucket: str
    rebuilt_key: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManifestEntry:
        return cls(
            ci_id=raw["ci_id"],
            lg=raw["lg"],
            year=int(raw["year"]),
            len_chars=int(raw["len_chars"]),
            ocrqa=float(raw["ocrqa"]),
            provider=raw["provider"],
            alias=raw["alias"],
            rebuilt_bucket=raw["rebuilt_bucket"],
            rebuilt_key=raw["rebuilt_key"],
        )

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SampleStats:
    """Summary stats over the sampled records for one language."""

    n: int
    year_min: int
    year_max: int
    year_mean: float
    ocrqa_min: float
    ocrqa_mean: float
    len_chars_min: int
    len_chars_max: int
    len_chars_mean: float


@dataclasses.dataclass
class SelectionStats:
    """Counters reported per run; surfaced in the run log and notes."""

    total_seen: int = 0
    not_article: int = 0
    lg_out_of_scope: int = 0
    provider_out_of_scope: int = 0
    year_out_of_window: int = 0
    ocrqa_below_min: int = 0
    too_short: int = 0
    too_long: int = 0
    eligible_per_lg: dict[str, int] = dataclasses.field(default_factory=dict)
    sampled_per_lg: dict[str, int] = dataclasses.field(default_factory=dict)
    sample_stats_per_lg: dict[str, SampleStats] = dataclasses.field(default_factory=dict)


def _summarise_sample(sampled: Sequence[ManifestEntry]) -> SampleStats:
    n = len(sampled)
    years = [e.year for e in sampled]
    ocrqas = [e.ocrqa for e in sampled]
    lens = [e.len_chars for e in sampled]
    return SampleStats(
        n=n,
        year_min=min(years),
        year_max=max(years),
        year_mean=sum(years) / n,
        ocrqa_min=min(ocrqas),
        ocrqa_mean=sum(ocrqas) / n,
        len_chars_min=min(lens),
        len_chars_max=max(lens),
        len_chars_mean=sum(lens) / n,
    )


def _parse_source_key(source_key: str) -> tuple[str, str, str]:
    """Extract ``(provider, alias, filename)`` from the aggregated source_key.

    Aggregator emits keys shaped like
    ``langident/langident-lid-ensemble_multilingual_v2-0-2/<provider>/<alias>/<alias>-<year>.jsonl.bz2``
    so we slice from the tail to be robust to prefix changes.
    """
    parts = source_key.split("/")
    if len(parts) < 3:
        raise ValueError(f"unexpected source_key shape: {source_key!r}")
    return parts[-3], parts[-2], parts[-1]


def _rebuilt_key_for(provider: str, alias: str, year: int) -> str:
    return f"{provider}/{alias}/{alias}-{year}.jsonl.bz2"


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield aggregated records from the local jsonl file.

    The aggregator emits plain ``.jsonl`` (not bz2). We call ``orjson.loads``
    directly to match the production codec choice (see io-throughput notes).
    A byte-scaled tqdm bar tracks progress through the (~30 GB) file;
    ``disable=None`` makes it auto-suppress under non-TTY stdout (pytest, CI).
    """
    total_bytes = path.stat().st_size
    with path.open("rb") as fh, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"scanning {path.name}",
        disable=None,
    ) as bar:
        for line in fh:
            bar.update(len(line))
            if not line.strip():
                continue
            yield orjson.loads(line)


def select_corpus(cfg: SelectionConfig) -> tuple[list[ManifestEntry], SelectionStats]:
    """Filter, sample, and return the manifest plus stats.

    A single streaming pass collects per-language eligible candidates; the
    sample is then drawn deterministically using ``cfg.seed`` so the manifest
    is reproducible. Memory cost is bounded by the eligible-set size (long,
    high-OCR articles in 3 languages × curated providers × 100-year window),
    which fits in RAM even at billions of input rows.
    """
    rng = random.Random(cfg.seed)
    languages = set(cfg.languages)
    # ``None`` (YAML ``null``) or empty list means "no provider filter for
    # this language": every provider is accepted. A non-empty set is the
    # curated allow-list.
    providers_by_lg: dict[str, set[str] | None] = {}
    for lg in languages:
        raw = cfg.providers.get(lg)
        providers_by_lg[lg] = set(raw) if raw else None
    eligible: dict[str, list[ManifestEntry]] = defaultdict(list)
    stats = SelectionStats()

    for record in _iter_records(cfg.input_path):
        stats.total_seen += 1

        if record.get("tp") != ARTICLE_TP:
            stats.not_article += 1
            continue

        lg = record.get("lg")
        if lg not in languages:
            stats.lg_out_of_scope += 1
            continue

        try:
            year = int(record["year"])
        except (KeyError, TypeError, ValueError):
            stats.year_out_of_window += 1
            continue
        if not (cfg.year_min <= year <= cfg.year_max):
            stats.year_out_of_window += 1
            continue

        ocrqa = record.get("ocrqa")
        if ocrqa is None or ocrqa < cfg.ocrqa_min:
            stats.ocrqa_below_min += 1
            continue

        len_chars = record.get("len")
        if not isinstance(len_chars, int) or len_chars < cfg.char_threshold(lg):
            stats.too_short += 1
            continue
        ceiling = cfg.char_ceiling(lg)
        if ceiling is not None and len_chars > ceiling:
            stats.too_long += 1
            continue

        try:
            provider, alias, _ = _parse_source_key(record.get("source_file") or record["source_key"])
        except (KeyError, ValueError):
            stats.provider_out_of_scope += 1
            continue
        allowed = providers_by_lg[lg]
        if allowed is not None and provider not in allowed:
            stats.provider_out_of_scope += 1
            continue

        ci_id = record.get("id")
        if not isinstance(ci_id, str):
            continue

        eligible[lg].append(
            ManifestEntry(
                ci_id=ci_id,
                lg=lg,
                year=year,
                len_chars=len_chars,
                ocrqa=float(ocrqa),
                provider=provider,
                alias=alias,
                rebuilt_bucket=REBUILT_BUCKET,
                rebuilt_key=_rebuilt_key_for(provider, alias, year),
            )
        )

    manifest: list[ManifestEntry] = []
    for lg in cfg.languages:
        pool = eligible.get(lg, [])
        stats.eligible_per_lg[lg] = len(pool)
        if not pool:
            stats.sampled_per_lg[lg] = 0
            continue
        # Stable seeded shuffle: sort by ci_id first to make input order
        # independent of file iteration order, then shuffle once with the
        # shared RNG and take the first k. This gives monotonic growth —
        # raising ``n_per_lg`` from N to M (M > N) extends the previous
        # sample with M-N new docs without dropping any of the original N.
        # Random.sample(pool, k) does NOT have this property: different k
        # values produce overlapping-but-not-nested samples even with the
        # same seed.
        pool.sort(key=lambda e: e.ci_id)
        rng.shuffle(pool)
        k = min(cfg.n_per_lg, len(pool))
        sampled = list(pool[:k])
        sampled.sort(key=lambda e: (e.year, e.ci_id))
        manifest.extend(sampled)
        stats.sampled_per_lg[lg] = k
        stats.sample_stats_per_lg[lg] = _summarise_sample(sampled)

    return manifest, stats


def write_manifest(manifest: Iterable[ManifestEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for entry in manifest:
            fh.write(orjson.dumps(entry.to_jsonable(), option=orjson.OPT_APPEND_NEWLINE))


def _format_stats(cfg: SelectionConfig, stats: SelectionStats) -> str:
    lines = [
        f"total_seen={stats.total_seen}",
        f"not_article={stats.not_article}",
        f"lg_out_of_scope={stats.lg_out_of_scope}",
        f"year_out_of_window={stats.year_out_of_window}",
        f"ocrqa_below_min={stats.ocrqa_below_min} (cutoff={cfg.ocrqa_min})",
        f"too_short={stats.too_short} (min_tokens={cfg.min_tokens})",
        f"too_long={stats.too_long} (max_tokens={cfg.max_tokens})",
        f"provider_out_of_scope={stats.provider_out_of_scope}",
    ]
    for lg in cfg.languages:
        provs = cfg.providers.get(lg)
        provs_str = ",".join(provs) if provs else "all"
        lines.append(
            f"  {lg}: eligible={stats.eligible_per_lg.get(lg, 0)} "
            f"sampled={stats.sampled_per_lg.get(lg, 0)} "
            f"(threshold={cfg.char_threshold(lg)} chars, providers={provs_str})"
        )
        s = stats.sample_stats_per_lg.get(lg)
        if s is not None:
            lines.append(
                f"      year=[{s.year_min}..{s.year_max}] mean={s.year_mean:.0f} | "
                f"ocrqa min={s.ocrqa_min:.3f} mean={s.ocrqa_mean:.3f} | "
                f"len_chars=[{s.len_chars_min}..{s.len_chars_max}] mean={s.len_chars_mean:.0f}"
            )
    return "\n".join(lines)


def _parse_provider_overrides(values: Sequence[str] | None) -> dict[str, tuple[str, ...]]:
    """Parse ``--providers fr=LeTemps,BNF de=NZZ,SWA`` style overrides."""
    if not values:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--providers entry must look like 'lg=Prov1,Prov2', got {item!r}"
            )
        lg, raw = item.split("=", 1)
        out[lg.strip()] = tuple(p.strip() for p in raw.split(",") if p.strip())
    return out


def _parse_chars_per_token_overrides(values: Sequence[str] | None) -> dict[str, float]:
    if not values:
        return {}
    out: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--chars-per-token entry must look like 'lg=4.5', got {item!r}"
            )
        lg, raw = item.split("=", 1)
        out[lg.strip()] = float(raw)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="impresso-research-corpus-select",
        description=(
            "Sample a per-language manifest of long, high-OCR newspaper articles "
            "from the langident-aggregated jsonl, for the chunking-eval sweep."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional study YAML (e.g. configs/research/study-v1.yaml). "
            "When given, populates corpus.* defaults; per-flag CLI args "
            "still override on a field-by-field basis."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "path to the local langident-aggregated .jsonl file "
            "(default: corpus.input_path from --config)"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output manifest path (.jsonl) "
            "(default: <paths.local_root>/manifest.jsonl from --config)"
        ),
    )
    p.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help=f"target languages (default: {' '.join(DEFAULT_LANGUAGES)})",
    )
    p.add_argument(
        "--n-per-lg",
        type=int,
        default=DEFAULT_N_PER_LG,
        help=f"sample size per language (default: {DEFAULT_N_PER_LG})",
    )
    p.add_argument(
        "--ocrqa-min",
        type=float,
        default=DEFAULT_OCRQA_MIN,
        help=f"OCR-quality cutoff applied uniformly across languages (default: {DEFAULT_OCRQA_MIN})",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=(
            "minimum article length expressed in tokens; per-language char threshold "
            f"= min_tokens * chars-per-token[lg] (default: {DEFAULT_MIN_TOKENS})"
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "optional upper bound on article length in tokens; required for "
            "Study A (docs that fit in context) and similar bounded windows. "
            "Default: no upper bound."
        ),
    )
    p.add_argument(
        "--chars-per-token",
        nargs="+",
        default=None,
        metavar="lg=value",
        help=(
            "override the chars-per-token estimate for one or more languages, "
            f"e.g. --chars-per-token fr=4.5 de=3.5 (defaults: {DEFAULT_CHARS_PER_TOKEN})"
        ),
    )
    p.add_argument(
        "--providers",
        nargs="+",
        default=None,
        metavar="lg=Prov1,Prov2",
        help=(
            "override the curated provider set for one or more languages, "
            "e.g. --providers fr=LeTemps,BNF de=NZZ. Pass an empty value "
            "(--providers fr=) to disable the filter and accept all providers "
            "for that language."
        ),
    )
    p.add_argument(
        "--year-min",
        type=int,
        default=DEFAULT_YEAR_MIN,
        help=f"earliest publication year, inclusive (default: {DEFAULT_YEAR_MIN})",
    )
    p.add_argument(
        "--year-max",
        type=int,
        default=DEFAULT_YEAR_MAX,
        help=f"latest publication year, inclusive (default: {DEFAULT_YEAR_MAX})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed for the per-language sample (default: {DEFAULT_SEED})",
    )
    return p


def config_from_args(args: argparse.Namespace) -> SelectionConfig:
    """Build a :class:`SelectionConfig` from parsed CLI args.

    Precedence: CLI flag (if explicitly different from its argparse
    default) > study YAML (if ``--config`` given) > module-level
    ``DEFAULT_*``. The argparse defaults below preserve the
    pre-refactor behaviour when no ``--config`` is supplied; when one
    is, every field that still matches its default is taken from the
    YAML instead.
    """
    study_cfg = None
    if args.config is not None:
        # Lazy import keeps tests that don't exercise YAML loading from
        # paying for pyyaml + pydantic at module-load time.
        from impresso_text_embedder.research.study_config import load_study_config

        study_cfg = load_study_config(args.config).corpus

    def _pick(cli_val: object, default_val: object, cfg_val: object | None) -> object:
        # If the CLI matches the module-level default, prefer the YAML
        # value when one is available. Otherwise the CLI wins.
        if cli_val == default_val and cfg_val is not None:
            return cfg_val
        return cli_val

    languages = tuple(_pick(
        tuple(args.languages),
        DEFAULT_LANGUAGES,
        tuple(study_cfg.languages) if study_cfg else None,
    ))
    n_per_lg = int(_pick(
        args.n_per_lg, DEFAULT_N_PER_LG,
        study_cfg.n_per_lg if study_cfg else None,
    ))
    ocrqa_min = float(_pick(
        args.ocrqa_min, DEFAULT_OCRQA_MIN,
        study_cfg.ocrqa_min if study_cfg else None,
    ))
    min_tokens = int(_pick(
        args.min_tokens, DEFAULT_MIN_TOKENS,
        study_cfg.min_tokens if study_cfg else None,
    ))
    year_min = int(_pick(
        args.year_min, DEFAULT_YEAR_MIN,
        study_cfg.year_min if study_cfg else None,
    ))
    year_max = int(_pick(
        args.year_max, DEFAULT_YEAR_MAX,
        study_cfg.year_max if study_cfg else None,
    ))
    seed = int(_pick(
        args.seed, DEFAULT_SEED,
        study_cfg.seed if study_cfg else None,
    ))
    # max_tokens: argparse default is None, so CLI override wins; YAML
    # supplies the fallback.
    max_tokens = args.max_tokens
    if max_tokens is None and study_cfg is not None:
        max_tokens = study_cfg.max_tokens

    chars_per_token = dict(DEFAULT_CHARS_PER_TOKEN)
    if study_cfg is not None:
        chars_per_token = dict(study_cfg.chars_per_token)
    chars_per_token.update(_parse_chars_per_token_overrides(args.chars_per_token))

    # ``None`` / empty tuple = no provider filter for this language. Preserve
    # ``None`` through copy so the YAML wildcard semantics survive into the
    # in-memory config.
    providers: dict[str, tuple[str, ...] | None] = {
        k: (tuple(v) if v else None) for k, v in DEFAULT_PROVIDERS.items()
    }
    if study_cfg is not None:
        providers = {
            k: (tuple(v) if v else None) for k, v in study_cfg.providers.items()
        }
    providers.update(_parse_provider_overrides(args.providers))

    missing_cpt = [lg for lg in languages if lg not in chars_per_token]
    if missing_cpt:
        raise SystemExit(
            f"no chars-per-token entry for language(s) {missing_cpt}; "
            f"pass --chars-per-token {missing_cpt[0]}=<value>"
        )
    missing_prov = [lg for lg in languages if lg not in providers]
    if missing_prov:
        raise SystemExit(
            f"no provider list for language(s) {missing_prov}; "
            f"pass --providers {missing_prov[0]}=<Prov1,Prov2>"
        )

    # Resolve --input and --output: CLI > YAML > error. Both are local
    # paths only — corpus_select doesn't touch S3 — so they bypass the
    # staged_input / staged_output helpers used elsewhere.
    input_path = args.input
    if input_path is None and study_cfg is not None:
        input_path = Path(study_cfg.input_path)
    if input_path is None:
        raise SystemExit("--input is required (or pass --config to derive it)")
    output_path = args.output
    if output_path is None and args.config is not None:
        from impresso_text_embedder.research.study_config import (
            MANIFEST_FILENAME,
            load_study_config,
        )

        full_cfg = load_study_config(args.config)
        output_path = full_cfg.local_path(MANIFEST_FILENAME)
    if output_path is None:
        raise SystemExit("--output is required (or pass --config to derive it)")

    return SelectionConfig(
        input_path=input_path,
        output_path=output_path,
        languages=languages,
        n_per_lg=n_per_lg,
        ocrqa_min=ocrqa_min,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        chars_per_token=chars_per_token,
        providers=providers,
        year_min=year_min,
        year_max=year_max,
        seed=seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    log.info(
        "corpus-select start: input=%s output=%s languages=%s n_per_lg=%d "
        "ocrqa_min=%.2f min_tokens=%d max_tokens=%s year=[%d,%d] seed=%d",
        cfg.input_path,
        cfg.output_path,
        cfg.languages,
        cfg.n_per_lg,
        cfg.ocrqa_min,
        cfg.min_tokens,
        cfg.max_tokens,
        cfg.year_min,
        cfg.year_max,
        cfg.seed,
    )
    manifest, stats = select_corpus(cfg)
    write_manifest(manifest, cfg.output_path)
    log.info("manifest written: %s (%d entries)", cfg.output_path, len(manifest))
    log.info("selection stats:\n%s", _format_stats(cfg, stats))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
