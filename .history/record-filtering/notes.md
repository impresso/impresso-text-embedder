# Record-level filtering — decisions

Central reference for "which records don't get embedded, and why". All
filter-reason keys live in `src/impresso_text_embedder/embed.py:54-60`
and are bumped via `_bump` / `_bump_and_log`. Tallies ride through the
per-file `filter_counter: Counter[str]` created in
`pipeline._encode_to_local` (`pipeline.py:149`) and surface in the
per-file `done` INFO line via `telemetry.format_stats_line`
(`telemetry.py:143-172`) as:

```
skipped=<total> (reason1=<n1> reason2=<n2> ...)
```

## Filter reasons (current)

| key | level | trigger | log level |
|---|---|---|---|
| `content_type` | text / sentence / chunk | `record["tp"]` is set and not in `cfg.content_types` (allow-list reject) | DEBUG per record |
| `missing_content_type` | text / sentence / chunk | `record["tp"]` absent or `None` | **WARN on first occurrence per file**, DEBUG thereafter |
| `missing_id` | text / sentence / chunk | `record["id"]` empty/absent | DEBUG per record |
| `too_short` | text / chunk | `len(document_text) <= cfg.min_char_length` (default 400) | DEBUG per record |
| `no_sentences` | sentence | `record["sents"]` empty/absent | DEBUG per record |
| `sentence_too_short` | sentence | every sentence has `len(stripped) <= cfg.min_char_length` | DEBUG per record |
| `no_chunks` | chunk | chunker returns empty for a doc that passed `too_short` | DEBUG per record |

Non-filter tally key surfaced in the same clause: `long_doc_chunked`
(text level, telemetry only — record is *emitted*, not skipped).

## Drift from legacy (`main:lib/text_embedding_processor.py`, commit `a433970`)

The legacy code only had the text level. Compared to `compute_embeddings`
(`a433970:lib/text_embedding_processor.py:252-342`):

| filter | `a433970` | current | notes |
|---|---|---|---|
| `content_type` | strict: `tp not in args.content_type` → `skipped_type_<tp>` | split: `content_type` (allow-list reject) + `missing_content_type` (absent `tp`, WARN once) | Restores legacy strictness for absent `tp`; adds visibility. |
| `content_type` CLI choices | `choices=["ar","page"]` in argparse | no `choices=` (cli/create.py) | Silent acceptance of typos. Known gap. Fix deferred until someone trips on it. |
| `min_char_length` | `len > 400` | `len <= 400` skips | Equivalent (same boundary via negation). |
| text source | `data["ft"]` only; missing `ft` falls through to `short_texts` | `ft` else `rebuild_ft_from_offsets(sents)` | Rebuilt-corpus shards carry `ft`; lingproc-enriched shards carry only `sents`. Both now embeddable. |
| `missing_id` | not checked | new filter | Legacy would have written an empty `id` to output. |
| long-doc handling | mandatory fixed-token-window chunk + mean pool | opt-in via `--long-doc-strategy=chunk` (default), `truncate` opt-out | Same effective behavior; see `.progress/long-doc-chunking/notes.md`. |
| sentence / chunk levels | did not exist | three new filters (`no_sentences`, `sentence_too_short`, `no_chunks`) | New code path. |

## Decision: `missing_content_type` separate from `content_type`, WARN-once

### Change

`_content_type_skip_reason` (`embed.py`) distinguishes two cases:

```python
def _content_type_skip_reason(record, cfg):
    tp = record.get("tp")
    if tp is None:
        return FILTER_MISSING_CONTENT_TYPE
    if tp not in cfg.content_types:
        return FILTER_CONTENT_TYPE
    return None
```

Missing `tp` is tallied under its own reason and triggers a single
WARNING per file on first occurrence (subsequent records stay at DEBUG
like every other filter). Wired at all three levels via a shared
`_bump_and_log(..., warn_first=(reason == FILTER_MISSING_CONTENT_TYPE))`
helper.

### Why

1. **Restore parity with `a433970`.** Legacy `None not in ["ar"]` → True
   → skip under `skipped_type_None`. The migration loosened this
   (`return tp is None or tp in cfg.content_types` → missing `tp`
   passed). That is a silent data-quality regression.
2. **Surface drift, don't hide it.** Real Impresso shards carry `tp` on
   every content item (`ar`, `page`, `ad`, `image`, …). A record with no
   `tp` means either a malformed shard or an upstream schema change —
   both worth noticing. Bumping a counter alone is too quiet; the
   per-file `done` line is easy to miss in a year of log.
3. **Cost of one WARN per file is acceptable.** If every file had one
   stray record with missing `tp` we'd see one WARN per file, not per
   record. If that turns out to be universal in production we revisit
   (see "Trade-off" below).

### Why two reasons, not one

Merging them under `content_type` would make the `skipped=N (…)` clause
lie about the cause. Operators debugging a shard want to tell apart
"the allow-list filtered out 10k `ad` records" (expected) from "10k
records had no `tp` field" (probably a bug upstream). Two keys, one tiny
helper function — cheap to keep distinct.

### Rejected alternatives

- **Per-record WARN.** A shard with thousands of missing-`tp` records
  would flood the log and the INFO-level file would be unreadable.
- **Summary-only WARN in the `done` line.** The `done` line is INFO; a
  single inline "(WARN: missing_content_type=N)" would not stand out in
  grep. Better to emit a dedicated WARN record that scrollback shows and
  any filter-by-level downstream collector picks up.
- **`--allow-missing-content-type` CLI flag.** Premature. We don't know
  yet whether real shards ever legitimately omit `tp`; if they do, a
  flag with documented rationale beats "just edited the code" — but only
  once the need is demonstrated.
- **Error out on first missing `tp`.** Too aggressive: one malformed
  record should not lose a full year-shard of otherwise-valid content.

## Trade-off

If production shards routinely lack `tp` the WARN fires once per file —
intended signal for the first run, noise after. Mitigation path (ordered
cheapest → most invasive):

1. Demote to INFO (one-line change in `_bump_and_log`).
2. Track the count in `format_stats_line` only, drop the dedicated WARN.
3. Add a `--allow-missing-content-type` flag that silences both the WARN
   and the skip (records pass through, are embedded under whatever
   `ci_type=None` downstream).

None of these is worth building pre-emptively — the point of the WARN
is to tell us whether we need any of them.
