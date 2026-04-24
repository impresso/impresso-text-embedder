# Step 17 — validate-source-stats

Extends `impresso-embed-validate` with (a) source-backed statistics for
records that show up in the "missing in target" / "missing in produced"
buckets, and (b) a Rich-rendered output across *all* validate stats. Both
are diagnostic improvements; `ValidationReport.passed` and exit codes are
unchanged.

Prior step that set up the groundwork: step 7 (`validation-metric`), see
`../validation-metric/notes.md`. That step deliberately rejected Rich to
keep CI logs grep-able — step 17 reverses that decision while preserving
every substring contract the earlier tests locked in.

## Why

`impresso-embed-validate --target …` can today report e.g. *"17 records
missing in target"* and *"3 records missing in produced"*, but gives no
way to say *anything* about those records. In practice users see the
count and can't distinguish:

- the golden was regenerated from a subset of the input
- the create run filtered them out via `--min-char-length` /
  `--content-type`
- the source shard itself doesn't contain those `id`s (upstream drift)
- they're legitimately empty records that can't be embedded

Cross-referencing the missing ids against the **source input**
`.jsonl.bz2` surfaces all four cases directly from the data the embedder
actually consumed.

The Rich rewrite is orthogonal but bundled into the same step because
the new source-stats block needs structured rendering (tables, panels,
per-direction sections) and it would be strictly worse to add that using
string concatenation while leaving the rest of the renderer plain.

## Scope

In:
- New CLI flag `--source <path|s3://…>` on `impresso-embed-validate`.
  Same path-resolution semantics as `--target` and the positional path
  (local or S3, streamed via `iter_lines_from_path`).
- Source-backed stats for **both** missing directions (missing in target
  *and* missing in produced). One stats block per direction, only
  rendered when that direction has ≥1 missing id.
- Per-direction stats surfaced:
  - char-length: min / p50 / p90 / p99 / max + log-scale histogram
  - `lg` language distribution (count per code; `(missing)` bucket for
    absent)
  - `tp` content-type distribution (same `(missing)` bucket rule)
  - count that would fall below `--source-min-char-length` (default 400,
    matching `impresso-embed-create`'s default)
  - count of records with neither `ft` nor reconstructable `sents`
    (legitimate empties)
  - count of missing ids *not found* in the source at all (upstream
    drift signal)
  - 3 sample excerpts (ci_id + first 80 chars of reconstructed text)
- Rich rendering by default. `rich.console.Console(force_terminal=None)`
  auto-detects the TTY and strips ANSI + re-flows when piped, so no
  explicit `--plain` flag is needed.
- Legacy substring markers (`records_checked=`, `MISMATCH:`, `OK`,
  `FAIL`, `p50=`, `log10(distance)`, `worst drifts`, `mismatches`,
  `missing in target (N)`) preserved inside the Rich output so all
  current tests and downstream log parsers continue to work without
  changes.

Out:
- Embedding-side analysis of the missing docs (they have no embedding —
  that's the point).
- Tokenizer-based token counts — would drag the full HF stack into the
  validate path; char length is the ~20× cheaper proxy and is good
  enough to distinguish a 3-char record from a 50 kB feuilleton.
- Multi-source-file support (one source shard per invocation).
- JSON sidecar output. Nobody asked for it; adding a file-output
  contract to a currently stdout-only tool should wait until a second
  consumer materialises.
- Running the full `impresso-embed-create` filter predicate
  (content-type allow-list, `lingproc_path` presence, etc.) against
  each missing record. Validate doesn't know which flags the create run
  used, so any such replay would be a lie by approximation.
  Char-length + `tp` / `lg` tallies cover the dominant cases; the user
  can eyeball the rest from the sample excerpts.

## Source file schema

Same shape as the create-CLI input (see `CLAUDE.md` → Data contract →
Input record):

- `id` — canonical content-item id. **Important:** input uses `id`
  while the embedding output uses `ci_id`. Match by
  `source_record["id"] == mismatch.ci_id`.
- `tp` — content type; optional in anomalous shards.
- `lg` — language code; optional.
- `sents` — list of sentence dicts. Used to reconstruct text via
  `text.rebuild_ft_from_offsets` when `ft` is absent.
- `ft` — full text; optional.

Loaded via the same `iter_lines_from_path` helper used for `path` /
`target`. One streaming pass, indexed against the set of missing
`ci_id`s collected during comparison. Early-exit when every missing id
has been accounted for.

## Output layout

Per-direction panel (only when that direction has ≥1 mismatch):

```
╭─ missing in target (17 records) ────────────────────────────╮
│ counts                                                       │
│   total                 17                                   │
│   found in source       14                                   │
│   not in source          3                                   │
│   reconstructable       12                                   │
│   empty (no ft/sents)    2                                   │
│   below min_char 400     5                                   │
│                                                              │
│ char length             min=11  p50=820  p90=4.2k ...        │
│ length histogram        (log10 bars)                         │
│                                                              │
│ lg                      de=8  fr=5  (missing)=1              │
│ tp                      ar=13  page=1                        │
│                                                              │
│ samples                                                      │
│   GDL-1798-05-03-a-i0017   "Le conseil a délibéré hier…"     │
│   GDL-1798-05-03-a-i0042   ""                                │
│   GDL-1798-05-03-a-i0091   "Annonces"                        │
╰──────────────────────────────────────────────────────────────╯
```

Overall flow for a comparison run:

1. Header (panel: `produced → target`)
2. Errors (only if any)
3. Counts table (records/items checked, level, tol, max distance)
4. Mismatch breakdown
5. Distance percentiles + log-scale histogram
6. Worst drifts table
7. **New:** source-stats panels — one per direction, only when
   `--source` is passed *and* that direction has missing ids
8. Verdict (`OK` / `OK: all records within tol` / `FAIL`)

Structural-only runs (no `--target`) with `--source` set: emit a
single WARNING on stderr (`"--source ignored without --target;
structural mode has no missing-records analysis"`) and otherwise ignore
the source. Avoids a silent no-op.

## Data model

New in `validate.py`:

```python
@dataclass
class SourceStatsBlock:
    direction: MismatchKind            # MISSING_IN_TARGET / MISSING_IN_PRODUCED
    total: int                         # ids requested
    found_in_source: int
    not_in_source_ids: list[str]
    reconstructable: int               # sents present (with or without ft)
    empty: int                         # neither ft nor sents → "" text
    below_min_char: int                # len(text) < min_char_length
    char_lengths: list[int]
    lg_counts: Counter[str]            # "(missing)" bucket for absent lg
    tp_counts: Counter[str]            # same rule
    samples: list[tuple[str, str]]     # (ci_id, text[:80])

@dataclass
class SourceStatsAnalysis:
    min_char_length: int
    blocks: dict[MismatchKind, SourceStatsBlock]
```

`ValidationReport` gains one optional field:

```python
source_stats: SourceStatsAnalysis | None = None
```

`passed` is untouched — source stats are diagnostic.

## Implementation order

1. `validate.py`: no change to match logic. After
   `validate_against_target` returns the existing `ValidationReport`,
   caller (CLI) gathers the missing ids per direction from
   `report.mismatches`.
2. `validate.py`: new `collect_source_stats(source_path, report,
   min_char_length=400) -> SourceStatsAnalysis` that streams the source
   once, reconstructs text per record, and tallies the block fields
   above.
3. `cli/validate.py`: add `--source`, `--source-min-char-length`
   flags. When both `--target` and `--source` are passed, call
   `collect_source_stats` and attach to `report.source_stats` before
   rendering.
4. `cli/validate.py`: introduce a single `rich.console.Console` per
   invocation. Replace the existing `print(...)` calls with
   `console.print(...)` and use `rich.table.Table` /
   `rich.panel.Panel`. Keep the existing ASCII histogram helper
   (`_log_histogram`) — Rich doesn't need to own that, and the output
   is already visually fine; wrap it in a `Panel` for consistency.
5. Legacy substring markers remain as plain `console.print(text,
   highlight=False)` lines: the legacy `records_checked=… items_checked=…
   max_distance=…` line, the `MISMATCH:` one-liner after worst drifts,
   the `missing in {target,produced} (N)` count line, and the final
   `OK` / `FAIL` verdict.
6. `pyproject.toml`: add `rich>=13` to runtime deps.
7. Tests: keep all existing substring assertions; add new cases for the
   source-file path (see below).
8. `CLAUDE.md`: add `--source` to the validate CLI description and
   a "Decisions recorded" bullet summarising the Rich switch + the
   source-stats feature.
9. `.progress/plan.md`: mark step 17 `done` once all of the above lands.

## Rich / test compatibility

pytest's `capsys` captures stdout as a non-TTY stream. Rich's default
`Console(force_terminal=None)` detects that and suppresses ANSI, so
`capsys.readouterr().out` sees the plain text — the existing substring
tests (`"records_checked="`, `"MISMATCH:"`, `"p50="`, `"log10(distance)"`,
`"worst drifts"`, `"mismatches"`, `"missing in target (1)"`, `"OK"`,
`"FAIL"`) continue to match without any test-only flag.

One snag to watch for: `rich.table.Table` renders with Unicode box-drawing
by default (`box.HEAVY_HEAD`). That's fine for humans and doesn't
interfere with substring tests, but log-grep tooling in Impresso ops
should continue to lean on the preserved ASCII markers (which we emit
outside the tables) rather than the table cells.

## Rejected alternatives

- **Token-accurate length via the HF tokenizer.** Pulls the transformer
  stack into the validate path; validate is supposed to run cheaply
  anywhere (including CI boxes without GPU). Char length is cheap and
  correlates well enough for "why did this not make it to the golden"
  diagnosis.
- **Per-record JSON sidecar output.** No current consumer. Can be added
  later without breaking the current CLI contract.
- **Replay the exact filter predicate from `pipeline.py`.** Validate
  doesn't know which flags the create run used. Char-length + `tp` /
  `lg` breakdowns surface the dominant filter reasons; the 3 sample
  excerpts cover the rest by eye.
- **Drop legacy grep markers to simplify the renderer.** Forces a test
  rewrite and breaks downstream log parsers (reader scripts grep for
  `MISMATCH:` and `records_checked=`). One extra `console.print` per
  marker costs nothing.
- **Separate `impresso-embed-explain` CLI.** Would duplicate the
  produced/target parsing and indexing. Keeping it one CLI is simpler
  and matches how users already think about validation.
- **`--plain` flag.** Rich auto-detects TTY on its own; an explicit
  flag is dead weight given `capsys`-based tests already get plain
  output for free. Revisit only if a user reports ANSI escapes landing
  in a non-pipe context.

## Extension — source stats for above-tolerance records (shipped)

Part 1 left the natural follow-up *"what do the records that drifted
above tolerance look like?"* on the table. Part 2 adds a third
direction block keyed by `MismatchKind.VALUE` so a single validate run
surfaces "drifted" and "missing" signals side-by-side, reusing the
streaming scan and the existing panel renderer.

Final test suite: 256/256 green, ruff clean.

### Why extend in-place vs. a new CLI

- The ids are already collected during comparison
  (`_compare_text` / `_compare_items` emit `MismatchKind.VALUE` entries
  with `ci_id`, `distance`, `tol`). The source-file is already streamed
  and indexed. A separate CLI would duplicate both.
- The render is a drop-in third `Panel` between the worst-drifts table
  and the missing-summary block; no new entry point, no new argparse
  flag, no new test fixture skeleton.
- The Rich renderer already renders N direction panels in a loop;
  extending the loop to include `VALUE` when populated is one `if` line.

### Scope

In:
- `SourceStatsBlock` tallied for `MismatchKind.VALUE` ids with the same
  fields as the missing directions (found / not-in-source /
  reconstructable / empty / below-min-char / char-lengths / lg / tp /
  samples).
- Drift-aware sampling: for the VALUE block, `samples` is populated
  with the **top-N worst drifts** (by cosine distance, descending)
  instead of the first-N encountered. Renders the exact 3 source
  records that drove the biggest deltas, which is what the user wants
  to stare at.
- Per-sample `distance: float | None` surfaced alongside the excerpt.
  Replaces the `tuple[str, str]` sample shape with a small `Sample`
  dataclass — `distance=None` for the missing directions (which have
  no distance by construction), populated for VALUE.
- Record-granularity aggregation over item-level mismatches: a record
  with K over-tol sentence/chunk items counts once, with the **max**
  distance across its items used for sample ranking. Matches how the
  missing-direction block already collapses items to records.
- New Rich panel: `source analysis — above tolerance (N records)`,
  border style distinct from the missing-direction panels (suggest
  `border_style="red"` vs. the current `yellow`) so the eye
  distinguishes "drift" from "missing" at a glance.

Out (deferred):
- **Per-lg / per-tp mean-drift tables** (e.g. "French docs average
  1.2e-3 drift, German 3.4e-4"). Real diagnostic value, but adds a
  fresh rendering path. Open item below.
- **Length-vs-distance scatter.** Same rationale — valuable, but a
  correlation plot wants a richer UI than a Rich panel allows. If a
  user hits real drift and asks, add it then.
- **Per-sentence / per-chunk detail** inside sample excerpts. The
  record-level excerpt is enough to identify the document; drilling
  into *which* sentence drifted is a downstream query the user can
  run against the raw mismatches list.
- **Independent threshold for the VALUE block.** Could imagine
  `--source-drift-floor 1e-3` to only sample records above a second
  threshold, but `--tol` already owns "what counts as drift" — adding
  another knob is premature.

### Data model

`SourceStatsBlock` gains:

```python
@dataclass
class Sample:
    ci_id: str
    excerpt: str
    distance: float | None = None  # populated for MismatchKind.VALUE only
```

Field change: `samples: list[tuple[str, str]]` → `list[Sample]`.
Breaking for the test suite — 2 existing tests unpack the tuple and
will be updated to `sample.ci_id`, `sample.excerpt`. Type is cleaner
than `tuple[str, str, float | None]` and self-documenting.

`SourceStatsAnalysis.blocks` now carries three `MismatchKind` keys
(`VALUE`, `MISSING_IN_TARGET`, `MISSING_IN_PRODUCED`). Empty `VALUE`
block still materialised (for symmetry with the missing directions)
but not rendered when `total == 0`.

### collect_source_stats changes

- `_missing_ids_by_direction` → rename to `_target_ids_by_direction`
  (or fold into the call site); extend to also extract `VALUE` ids.
  Keep record-granularity: a single `ci_id` with several item-level
  mismatches shows up once, and the per-record **max distance** across
  its items is tracked separately for sample ranking.
- New helper `_value_distance_per_record(mismatches) -> dict[str, float]`
  returning `{ci_id: max_distance}`. Feeds sample ordering.
- Sampling logic changes shape:
  - Missing directions: keep the current "first N seen in source"
    behaviour (order is arbitrary; the stats are the point).
  - VALUE direction: buffer *all* matched records, then pick the
    top-N by distance descending at finalise time. For typical shard
    sizes (thousands of records, dozens above-tol) the extra
    bookkeeping is trivial.
- `Sample.distance` filled in for VALUE entries from
  `_value_distance_per_record`; `None` for missing entries.

### CLI render changes

- `_render_source_stats_block` accepts the new `Sample` and emits the
  distance next to the excerpt when present:
  `  {ci_id}  d={dist:.2e}  {excerpt!r}`
  For missing-direction panels the `d=…` segment is omitted (no
  distance).
- `_emit_source_stats` iterates over the three directions in order
  `VALUE` → `MISSING_IN_TARGET` → `MISSING_IN_PRODUCED` so a reader
  scanning top-to-bottom sees "what drifted" before "what's missing".
- Panel title: `"source analysis — above tolerance (N records)"`;
  border_style `red` so it visually outranks the yellow missing
  panels.

### Test plan

New cases in `tests/test_validate.py::TestSourceStats`:

- `test_value_direction_populated` — triple with a single above-tol
  record; source has it; assert VALUE block has `total=1`,
  `found_in_source=1`, one sample with populated `distance`.
- `test_value_samples_ordered_by_worst_drift` — 10 above-tol records
  with monotonically-increasing distances; assert
  `[s.ci_id for s in block.samples]` matches the 3 worst by distance.
- `test_value_items_collapse_to_record_max_distance` — one ci_id with
  3 item-level mismatches at distances `[0.01, 0.5, 0.02]`; assert
  the sample distance for that record is `0.5`.
- `test_sample_dataclass_migration` — existing "tuple unpack"
  assertions replaced with `sample.ci_id` / `sample.excerpt`.

New case in `tests/test_cli_validate.py`:

- `test_above_tol_panel_renders` — produce a comparison with 2
  above-tol records, `--source` set; assert
  `"above tolerance"` substring appears in `out` (panel title),
  `"d="` appears for the rendered sample line.

### Backwards compat

- `SourceStatsBlock.samples` shape changes from
  `list[tuple[str, str]]` → `list[Sample]`. Public data-layer change.
  Impact: the 3 existing tests that unpack the tuple need a one-line
  edit each; no external consumers depend on this field (validate
  writes no sidecar).
- CLI output gains the VALUE panel — strictly additive for users who
  aren't reading the panels programmatically. No legacy substring is
  removed; all step-7 and step-17-part-1 markers stay.

### Rejected alternatives

- **Separate `DriftStatsBlock` / separate `DriftAnalysis` type.**
  Structurally pure but forces a fork in the renderer and a fork in
  `collect_source_stats`. Same fields with one extra `distance` per
  sample is simpler.
- **Surface above-tol samples inside the existing `worst drifts`
  table** (as a fourth column with source excerpts). Tempting because
  the worst-drift table already orders by distance. Rejected: it
  would duplicate the source-lookup logic, and the worst-drift table
  is a compact numeric view the renderer would rather keep tight.
  Separate panel buys a full counts table + histogram + lg/tp
  breakdown that a table cell can't.
- **Thresholded per-bucket sampling** (sample the worst record from
  each char-length bucket). Smarter signal per record, but burns user
  attention trying to understand the bucketing scheme. Three
  top-drift samples is easier to reason about.

### Open items (extension-specific)

- **Per-lg / per-tp mean-drift tables.** Would answer "does X% of
  drift live in one language?" without leaving the panel. Deferred
  until someone hits real drift and wants the signal.
- **Drift vs length correlation.** Could compute
  `pearson(char_length, distance)` inline and print one line. Cheap,
  but adds a stats dep (or we hand-roll). Revisit if the per-lg /
  per-tp tables ship.
- **Sample count knob.** Currently hard-coded to 3. If the VALUE
  panel becomes the primary use of validate, a `--source-samples N`
  flag may earn its keep.

## Open items (post-implementation)

- **Length histogram edges.** Default to powers of 10
  (`1, 10, 100, 1k, 10k, 100k`). Revisit if real Impresso long-doc data
  clusters badly (e.g. everything in `100–1k`).
- **Sample count.** 3 per direction. Surface as a flag only if a user
  actually asks.
- **`--source-min-char-length` default.** Kept at `400` because that's
  `impresso-embed-create`'s default. If a run used a different
  threshold, the "below min_char" count in the validate output will be
  off by that much — acceptable; the count is a pointer, not a proof.
  Could be auto-derived from the create run's own log one day.
- **Missing in produced *and* target simultaneously.** Currently
  impossible by construction (the union of indexed ids splits the two
  directions cleanly). If a future schema allows partial records this
  assumption needs revisiting.
