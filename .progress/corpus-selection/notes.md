# corpus-selection — design notes

Design narrative for step 1 of `research/chunking-eval`: building a stable
per-language manifest of long, high-OCR newspaper articles that the
downstream chunking-strategy sweeps will operate on.

## Source

Local copy of the Impresso langident-aggregated jsonl:
`tmp/115-canonical-processed-final-langident-langident-lid-ensemble_multilingual_v2-0-2__AGGREGATED.jsonl`
(30 GB, 66,518,904 records). One record per content item, schema:

```
{id, year, lg, len, lg_decision, tp, alphabetical_ratio, ocrqa, source_file, source_bucket, source_key}
```

`source_key` is shaped
`langident/langident-lid-ensemble_multilingual_v2-0-2/<provider>/<alias>/<alias>-<year>.jsonl.bz2`,
which is what the manifest parses to recover provider/alias/year. The
rebuilt-text path that downstream steps fetch is `122-rebuilt-final/<provider>/<alias>/<alias>-<year>.jsonl.bz2`.

## Decisions (Q&A frozen from the session that produced this step)

| # | Question | Decision | Rationale / consequence |
|---|---|---|---|
| Q1 | what does `len` measure? | **character count** | Impresso convention, confirmed by the user. All token thresholds in this step convert via per-language chars-per-token. |
| Q2 | length threshold | **min-tokens=4000, per-lg chars-per-token (fr=4.5, de=3.5, lb=3.5)** → fr≥18000 chars, de/lb≥14000 chars | Smaller windows {512,1024,2048,4096} all chunk multiple times; 8190 stays one-shot; the window that the boundary question hinges on. |
| Q3 | OCR cutoff | **uniform `ocrqa>=0.9` across all languages, parameterised** | "Higher is better, same for every language." Consequence: lb has zero eligibles — see below. |
| Q4 | per-language target N | **200, parameterised** (`--n-per-lg`) | Matches Chroma's chunking benchmark scale. Sampling is a **stable seeded shuffle** so raising `--n-per-lg` extends the previous sample monotonically — see "Sampling: stable seeded shuffle" below. |
| Q5 | provider/topic bounding | **curated providers per lg, derived from a 1/50 stratified scan, parameterised** (`--providers`) | Topic bound + LLM cost cap. The lists were corrected after the first run (NZZ/SWA/SUB carry almost no de articles — SNL is the dominant multilingual carrier). |
| Q6 | year window | **1880–1980 inclusive, parameterised** (`--year-min/--year-max`) | Peak content density; pre-1880 trickle and post-1980 sparse tail dropped. |
| Q7 | manifest deliverable | **`.jsonl` with `{ci_id, lg, year, len_chars, ocrqa, provider, alias, rebuilt_bucket, rebuilt_key}`** | Self-contained pointer to `122-rebuilt-final` for the next step. |

## Chosen parameter set

```
languages       = ("fr", "de")
n_per_lg        = 200
ocrqa_min       = 0.90
min_tokens      = 4000
chars_per_token = {fr: 4.5, de: 3.5, lb: 3.5}
providers       = {fr: (BNF, LeTemps, SNL, FedGaz),
                   de: (SNL, FedGaz, BNL),
                   lb: (BNL,)}     # listed but not used at default ocrqa
year window     = [1880, 1980]
seed            = 42
```

Output: `tmp/chunking-eval/corpus-manifest.jsonl` (400 entries, 200 fr + 200 de).

## Eligible-count sweep (full corpus, ocrqa≥0.9, 1880–1980, articles, curated providers)

Numbers from a 1/50 stratified scan, multiplied by 50 for the full-corpus estimate:

| lg | ≥1000 tok | ≥2000 | ≥3000 | **≥4000** | ≥5000 | ≥6000 | ≥8000 |
|---|---|---|---|---|---|---|---|
| fr | 1,575,450 | 357,600 | 92,700 | **31,850** | 13,150 | 7,700 | 4,050 |
| de | 252,250 | 86,800 | 38,300 | **20,200** | 8,450 | 4,150 | 1,550 |
| lb | 0 | 0 | 0 | **0** | 0 | 0 | 0 |

The full-file selection run produced fr=32,150 / de=19,734 eligibles —
within ~1% of the projection.

## Sampling: stable seeded shuffle

The per-language sampler sorts the eligible pool by `ci_id` (so input order
is independent of file iteration), then `rng.shuffle(pool)` once with the
shared `random.Random(seed)`, then takes `pool[:n_per_lg]`. This gives the
**monotonic-growth** property: for any `M > N`, the sample at
`--n-per-lg=M` is a strict superset of the sample at `--n-per-lg=N` (with
all other parameters held). Verified at corpus scale: a 600-doc run is a
strict superset of the 400-doc run, both fr and de.

This is *not* the property of `random.Random.sample(pool, k)`, which we
were using initially — `sample(pool, 200)` and `sample(pool, 300)` produce
overlapping-but-not-nested sets even with the same seed, because Python's
sample algorithm draws each new k from scratch rather than slicing a
shared shuffle.

### Why a stable shuffle, not "top-N by ocrqa"

Both schemes give the monotonic property; we picked stable shuffle because
within the filtered pool the OCR variance is small (fr 0.90–0.99, de
0.90–0.95) and a top-N-by-ocrqa picker would narrow representativeness.
Specifically, "top by OCR" correlates with shorter, more recent,
easier-to-recognise documents, which would amplify the de pre-1940
underrepresentation we already have (see O6) and shrink length diversity.
Within-band OCR differences are dominated by tokenizer/chunker effects in
the eval, not by upstream OCR noise — sorting by ocrqa is mostly noise.

Hybrid bucket-then-shuffle (D in the discussion) was rejected as overkill
for this step. Score-based ranking remains available as a future knob if
the eval surfaces evidence that within-band OCR variation contaminates
the chunking signal.

## Why lb is dropped from the default sweep

A targeted scan of every article (`tp=article, lg=lb, year ∈ [1880,1980]`)
in a 1/50 stratified sample (1,193 articles representing ~60k full-corpus):

- **All-articles ocrqa**: p25=0.67, p50=0.74, p75=0.81, p95=0.89, max=0.97
- **`len ≥ 14000 chars`**: 2 articles in 1/50 sample (~100 in full corpus), **all with ocrqa 0.65–0.67**

i.e. the long lb articles in the corpus are systematically the worst-OCR
ones — likely because long articles span more page area and accumulate more
recognition errors, and lb publications skew toward earlier print quality.
There is no `(length, ocrqa)` cell where lb provides a sample at the
quality bar the other languages hit. This is a **corpus property**, not a
filter mistake.

Consequence: dropping lb from `DEFAULT_LANGUAGES` is the honest move.
`lb` remains in `DEFAULT_PROVIDERS` and `DEFAULT_CHARS_PER_TOKEN` so
opt-in via `--languages fr de lb --ocrqa-min 0.65` still works for a
sensitivity / cross-lingual side experiment.

## Provider-list correction (the foot-gun avoided)

First-pass guess based on newspaper recognisability — `fr=(LeTemps, BNF, BCUL)`,
`de=(NZZ, SWA, SUB)`, `lb=(BNL,)` — produced fr=19,516 eligibles but
**de=0**. The 1/200 stratified scan revealed why: NZZ/SWA/SUB barely
surface in the article-level partition of the corpus (they're either
predominantly page-level or under-represented in this aggregate). The
real long+clean de carriers are **SNL**, then FedGaz and BNL. SNL is
also the dominant multilingual carrier on the fr side (it's the Swiss
National Library aggregator). The defaults were corrected and the
re-run produced fr=32,150 / de=19,734.

**Generalisation**: don't pick providers by "what newspaper sounds
familiar". Always run a per-`(lg, length, ocrqa)` provider histogram first
and pick the carriers that empirically supply the long+clean tail.

## Sample composition (output manifest)

**fr (n=200)**
- Decades: 1880=41, 1890=40, 1900=24, 1910=17, 1920=21, 1930=17, 1940=11, 1950=10, 1960=7, 1970=12 — smooth across the window.
- Providers: BNF=102, FedGaz=38, LeTemps=30, SNL=30.
- 16 unique aliases (top: jdpl=55, FedGazFr=38, legaulois=24, LSE=22, JDG=15).
- Length (chars): min=18,008, p50=21,074, p90=35,899, max=310,847.
- OCR: min=0.90, p50=0.93, max=0.99.

**de (n=200)**
- Decades: 1890=1, 1900=3, 1910=8, 1920=4, 1930=6, 1940=28, 1950=64, 1960=60, 1970=26 — **skewed late**, only 22 articles before 1940.
- Providers: SNL=154, FedGaz=37, BNL=9.
- 10 unique aliases (top: DTT=120, FedGazDe=37, FZG=31, luxland=4, luxwort=2).
- Length (chars): min=14,042, p50=16,901, p90=24,730, max=379,969.
- OCR: min=0.90, p50=0.91, max=0.95.

The de decade skew is a direct consequence of the OCR cutoff: pre-1940
de print is poorer-quality and gets filtered out. Left uncorrected for
this step — the chunking question is about tokens-vs-attention, not
historical period. If a per-decade balance turns out to matter for the
eval, the fix is a stratified-by-decade sampler at this layer, not a
relaxation of `ocrqa_min`.

## Rejected alternatives

- **Per-language ocrqa cutoffs to "save" lb** (e.g. `lb: ocrqa>=0.65`).
  Rejected: long lb articles available at any ocrqa <0.7 are by
  construction noisy; the eval metric would partly measure OCR
  robustness rather than chunking strategy. The clean answer is "the
  corpus does not support this experiment for lb at the chosen quality
  bar" — that's a real finding, not an obstacle.
- **Drop length threshold for lb only** (e.g. `lb: min_tokens=1000`).
  Rejected: still 0 eligibles at `ocrqa>=0.9` for short lb articles too
  — the providers that carry lb articles don't reach the OCR bar at any
  length. Even if it gave nonzero pool, the methodology would no longer
  be comparable across languages (smaller documents → fewer chunks at
  every window, different signal).
- **Tokenise during selection to avoid the chars-per-token approximation**.
  Rejected for now: the approximation is generous (4.5 fr / 3.5 de) so
  some borderline articles get included that wouldn't pass at exact
  4000 tokens — fine, the downstream chunker re-tokenises authoritatively.
  Refining `chars_per_token` from a real tokenisation pass on the
  selected manifest is the chars_per_token-calibration follow-up step.
- **`random.sample(pool, k)` for the per-language pick.** Rejected: not
  monotonic across `k`. Bumping `--n-per-lg` from N to M reshuffles the
  whole sample, which would defeat reproducibility when we want to add
  more docs to an existing run. Replaced by the stable seeded shuffle
  documented above.
- **Top-N by ocrqa.** Monotonic and "best by quality" sounds defensible,
  but rejected — see "Why a stable shuffle, not 'top-N by ocrqa'"
  above. Within-band variance too small, narrows representativeness.
- **Auto-derive curated providers from the long+clean histogram** (skip
  the manual `DEFAULT_PROVIDERS` table). Rejected: the lists need a
  human in the loop because "topic bounding" is partly editorial — we
  want some diversity (multiple newspapers per language) but not so
  much that the LLM query-generation prompt has to handle every Swiss
  cantonal weekly. The histogram informs the choice, doesn't replace it.
- **Sample uniformly across providers (equal docs per provider)**.
  Rejected: provider sizes vary by orders of magnitude; uniform
  sampling would over-represent thin providers and miss the dominant
  carriers' editorial style. Random sampling proportional to
  eligible-set size is the default; a stratified knob is a deferred
  follow-up if the evaluation reveals provider-effect contamination.
- **`tp=ar` filter** (per CLAUDE.md production convention). Rejected:
  the aggregator emits the long form `tp=article`, not the rebuilt-side
  short form `tp=ar`. The selection module hard-codes `ARTICLE_TP =
  "article"` to match the aggregator schema; the downstream rebuilt-text
  fetcher will still get records tagged `ar` and that's fine — different
  schemas, same content.

## Open items

- **chars_per_token calibration** — the 4.5/3.5 defaults are
  conservative literature estimates. Once the manifest's text is
  fetched, run the gte-multilingual-base tokenizer over it and emit
  per-language empirical chars-per-token; revise this step's threshold
  if the gap is >10%. Listed as a separate research deliverable in the
  branch scope.
- **de pre-1940 underrepresentation** — only 22 de articles before
  1940 in the manifest. Decide later whether to add a stratified-by-
  decade sampler or accept it.
- **Cross-lingual ablation feasibility** — gated on the monolingual
  sweep. If lb resurfaces it'll need the relaxed-ocrqa workaround.
- **Reproducibility check** — running the CLI a second time with the
  same seed produces the same 400 ci_ids. Verified by a unit test
  (`test_sample_is_deterministic_across_runs`), not yet by a full re-run
  on the 30 GB file.
- **Monotonic-growth check** — running with `--n-per-lg=300` produces a
  strict superset of the `--n-per-lg=200` manifest. Verified at unit
  scale (`test_increasing_n_per_lg_extends_previous_sample`) and at
  corpus scale (manual: 600-doc run ⊃ 400-doc run, both fr and de).

## Reproducing

```
uv run python -m impresso_text_embedder.research.corpus_select \
  --input  tmp/115-canonical-processed-final-langident-langident-lid-ensemble_multilingual_v2-0-2__AGGREGATED.jsonl \
  --output tmp/chunking-eval/corpus-manifest.jsonl
```

(~65s on a Mac for the full 30 GB file, single pass, no parallelism
needed.) Defaults are wired to the parameter set above; pass any of
`--languages / --n-per-lg / --ocrqa-min / --min-tokens /
--chars-per-token / --providers / --year-min / --year-max / --seed` to
override.
