# aggregation-sweep — design notes

Step 10 of `research/chunking-eval`: introduce a new study
`C-aggregator` that compares all four registered aggregation
strategies (`mean`, `max`, `first-chunk`, `length-weighted`) against
the `token-budget` chunker at three sizes `[256, 1024, 4096]`. Reuses
study-A-fit's corpus / queries / queries-embedded artefacts via a new
server-side S3 copy CLI so the seven prior Run:AI jobs of corpus and
query work do not have to repeat.

## Why this step exists

[`plan.md` § Scope](../plan.md#scope-locked-with-the-user) committed
the team to "Length-weighted / max / first-chunk ablations only on
the winning chunker" as a follow-up to the primary chunker sweep. We
are realising that line early, on `token-budget` — the chunker the
user picked — rather than gating on a definitive winner from
study-A-fit, because:

- The four aggregator modules (`mean`, `max`, `first-chunk`,
  `length-weighted`) are already implemented, registered, and tested
  (prep commit `2e46896`).
- `embed.py:380-398` already passes per-chunk token counts as
  `weights` to the aggregator, so `length-weighted` works end-to-end
  with no encoder change.
- Running it in parallel with the chunker sweep doubles the
  information per Run:AI cycle without doubling corpus or query work
  (the seed CLI does the heavy reuse).

The only thing missing from the existing pipeline was a way to express
multiple aggregators in one study config. The previous schema collapsed
aggregation to a single global value, which would have forced four
separate studies — four corpus copies, four query copies, four
`config_sha` values that defeat apples-to-apples comparison.

## Scope (locked with the user)

- **One study, one aggregator dimension** — extend the cartesian to
  `chunkers × chunk_sizes × aggregators` rather than running four
  parallel single-aggregator studies.
- **Backward-compat with the singular form** — `aggregator: mean` (the
  legacy spelling) keeps working bit-identically; `aggregators:
  [mean, max, ...]` is the new opt-in. Setting both is a schema error.
- **Singleton path is byte-identical** to pre-change behaviour. A-fit /
  B-overflow / v1 keep their scenario IDs, labels, and `config_sha` —
  enforced by the regression test
  `test_build_scenarios_v1_grid_matches_legacy_layout`.
- **Server-side S3 copy** for the seed (no body transfer), with idempotent
  skip-if-exists. Same Ceph bucket only — cross-bucket copies are out
  of scope.
- **Output records carry `aggregator`** so downstream eval can slice
  by it. `(none)` for the truncate baseline.

## Mechanism

### Schema (`research/study_config.py`)

`ScenariosConfig` gains:

```python
aggregators: tuple[str, ...] | None = None

@model_validator(mode="after")
def _aggregator_xor_aggregators(self): ...

def effective_aggregators(self) -> tuple[str, ...]:
    return self.aggregators if self.aggregators is not None else (self.aggregator,)
```

Mutual exclusivity: `aggregators` set + `aggregator != "mean"`
(the default) → ValueError. The cleaner "exactly one of A, B" check
isn't possible because `aggregator` always has a default.
`effective_aggregators()` is the one place downstream code reads the
list from, so the singular/plural duality is invisible past this
boundary.

### Builder (`research/scenario_builder.py`)

```
for chunker in cfg.chunkers:
    for size in cfg.chunk_sizes:
        for agg in cfg.effective_aggregators():
            ...
```

Aggregator is the **innermost** loop. Two reasons:

1. When the singleton form is used (`aggregators is None`), the inner
   loop runs once and the IDs / labels match the pre-change
   sequence bit-for-bit.
2. When multiple aggregators are listed, "all aggregators at one
   (chunker, size) cell" is a contiguous slice of IDs — the smallest
   cell can be smoke-tested by running `S1..S{|aggs|}` first.

Label rule: `f"{chunker}-{size}-{agg}"` only when `len(aggs) > 1`,
else legacy `f"{chunker}-{size}"`. Locks backward compat.

### S3 copy (`io.py` + `research/study_seed.py`)

`copy_s3_object(src_bucket, src_key, dst_bucket, dst_key, *,
overwrite=False) -> bool`:

1. If not `overwrite` and dst exists → log+return False (idempotent).
2. HEAD src (cheap; gives source size for parity check).
3. `s3.copy_object(CopySource={...}, Bucket=dst_bucket, Key=dst_key)`
   — server-side, no body transfer.
4. HEAD dst, assert size matches src.

The seed CLI wraps this for the three conventional artefact filenames
(`corpus`, `queries`, `queries-embedded`) and resolves S3 keys via
`StudyConfig.s3_key()`. `--dry-run` prints the (src, dst) pairs and
returns 0 without touching S3. `--overwrite` propagates to the helper.

### Output schema

`embed_sweep._build_output_record` adds:

```python
enriched["aggregator"] = scenario.aggregator_name or "(none)"
```

`(none)` rather than `None` so downstream `groupby` / Pandas
operations don't surprise on null handling. Truncate baselines (S0)
get the literal string `(none)`.

## Rejected alternatives

- **Run four separate studies, one per aggregator.** Rejected: 4×
  S3 storage for corpus/queries/queries-embedded, 4× query-embed
  cost, 4 different `config_sha` values that defeat apples-to-apples
  comparison in the eval. The seed CLI was cheaper to write than
  rationalising those duplicates after the fact.
- **Fully explicit `scenarios:` list in YAML** (e.g.
  `scenarios: [{chunker: token-budget, size: 1024, aggregator: max}, ...]`).
  Rejected: breaks the "the grid is a cartesian product" invariant
  the rest of the system commits to. The Makefile shells out to
  `--list-ids`, ScenarioRegistry treats scenarios as a flat
  enumeration, and `n_scenarios()` is a multiplication. Adding a
  second code path for hand-written scenarios doubles the surface
  area for one corner case.
- **Aggregator-major ordering** (`for agg: for chunker: for size`).
  Rejected: makes the singleton path non-bit-identical to the
  pre-change layout, which would have forced regenerating
  A-fit/B-overflow `config_sha` and re-running everything.
- **One-off `scripts/seed_study.py` instead of a CLI.** Rejected:
  buries the capability where it's not pip-installable and not
  unit-testable; the helper in `io.py` is reusable beyond this study
  (e.g. promoting a sandbox artefact to a frozen prefix later).
- **Cross-bucket S3 copies.** Rejected: would need extra IAM
  reasoning and is out of scope for the current "one bucket, one
  prefix" research layout. Hard fail at CLI parse time when src and
  dst buckets disagree.
- **`copy_object` ETag check instead of size check** for post-copy
  verification. Rejected: server-side copies on Ceph can change the
  ETag (multipart re-chunking); size is the floor that catches
  partial-copy failures without false positives on legitimate
  re-chunking.

## Open items

- O30 — Run the C-aggregator study end-to-end on RCP and write the
  per-aggregator Recall@k / Precision_Ω deltas vs S0 into this notes
  file. Per the locked scope, only the winning aggregator from this
  sweep proceeds to the cross-chunker ablation.
- ~~O31 — Extend the eval (step 8) to surface the new `aggregator`
  field as a stratification axis alongside `chunker` / `chunk_tokens`.~~
  Closed: `score_queries` now emits an `aggregator` column;
  `aggregate_recall_with_ci` and `baseline_delta_table` default `by=`
  tuples include it. Singleton-aggregator studies (A-fit, B-overflow,
  v1) are unaffected — every row gets `"mean"` and the groupby cell
  count is unchanged. Multi-aggregator studies (C-aggregator) get the
  `chunk_size × aggregator` heatmap from one groupby.
- O32 — Decide whether `query-embed` outputs should also carry an
  `aggregator` field for symmetry. Today queries are never chunked
  (one short string, one vector), so the field would always be
  `(none)`. Probably skip — the asymmetry is meaningful (queries are
  not aggregated; docs are).
- O33 — Add a `make seed-study` Makefile convenience target wrapping
  `impresso-research-study-seed` if the pattern recurs for future
  studies (D-, E-, …). Single-use today.

## Verification

Unit tests added in the same diff:

- `tests/test_study_config.py`: parsing the `aggregators` field;
  mutual exclusivity vs `aggregator`; empty-list rejection;
  `n_scenarios()` accounts for the third dimension; loads
  `configs/research/study-C-aggregator.yaml` end-to-end.
- `tests/test_research_scenarios.py`: cartesian expansion arithmetic;
  size-major / agg-innermost order; label format switches only when
  plural; unknown aggregator name rejected; baseline still emits S0.
- `tests/test_io.py`: `TestCopyS3Object` covers happy path,
  skip-if-dst-exists, overwrite, post-copy size mismatch, missing
  source, copy_object failure.
- `tests/test_research_study_seed.py`: dry-run prints pairs without
  calling the helper; happy path invokes copy per artifact;
  `--overwrite` propagates; subset filter works; identical study
  names rejected; cross-bucket rejected.
- `tests/test_research_embed_sweep.py`: S0 emits `aggregator =
  "(none)"`; chunked scenarios emit the actual aggregator name; new
  end-to-end test runs `fixed-window-512-length-weighted` on a long
  doc and asserts the output is finite and L2-unit (regression lock
  for the weight-passing path in `embed.py:380-398`).

Smoke checks performed:

- `python -m impresso_text_embedder.research.scenario_builder
  --config configs/research/study-A-fit.yaml --list-ids` returns
  exactly the pre-change ID sequence.
- The same on `study-C-aggregator.yaml` returns 13 IDs S0..S12 with
  size-major / agg-innermost label ordering.
- `impresso-research-study-seed --source-config A-fit.yaml
  --target-config study-C-aggregator.yaml --dry-run` prints 3 pairs, no
  boto3 mutations.

568/568 tests pass on the full suite after the change.
