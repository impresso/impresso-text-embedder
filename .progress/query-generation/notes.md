# query-generation — design notes

Step 4 of `research/chunking-eval`: generate a synthetic
`(query, gold_excerpts)` set against the corpus shard from step 2 so
the eval step (TBD) can score every scenario from step 3 with
token-level Recall@k / IoU / Precision_Ω. Queries are
**position-stratified** (head / mid / tail third of each source doc) so
no chunker can win by silently dropping the back half of every
article.

## Inputs / outputs

- **Input**: the corpus shard built by step 2,
  `s3://140-processed-data-sandbox/chunking-eval/corpus/corpus-v1.jsonl.bz2`
  (overridable via `--corpus-bucket` / `--corpus-key`; `--local-corpus`
  skips the download for laptop iteration). Each record carries
  `{ci_id, lg, year, provider, alias, ocrqa, len_chars, tp, ft, sents}`.
- **Output**: a single `.jsonl.bz2` of queries at
  `s3://140-processed-data-sandbox/chunking-eval/queries/queries-v1.jsonl.bz2`
  (one record per *query*, not per doc).
- **Per-record schema**:

  ```
  {
    query_id,            // "<ci_id>__<bucket>__<k>"
    ci_id,
    lg,                  // mirrored from corpus
    query_text,
    query_type,          // "question" | "topical-phrase"
    references: [        // verbatim spans in the source ft, in-bucket
      {text, char_start, char_end},
      ...
    ],
    position_bucket,     // "head" | "mid" | "tail"
    position_chars: [bucket_lo, bucket_hi],
    gen_model,           // e.g. "Qwen/Qwen3-30B-A3B-Instruct-2507"
    gen_endpoint,        // e.g. "https://inference.rcp.epfl.ch/v1"
    ts                   // UTC YYYY-MM-DDTHH:MM:SSZ
  }
  ```

  Fields chosen so the eval step can compute Recall/IoU directly from
  `references[].char_start..char_end` against per-scenario chunk
  spans, and stratify by `position_bucket` × `lg` in one pass.

## Scope (locked with the user)

- **Endpoint**: EPFL RCP AIaaS only
  (`https://inference.rcp.epfl.ch/v1`, OpenAI-compatible, 5-parallel
  cap per API key). No CaaS fallback path; if AIaaS is unsuitable
  later we'll address it then. Auth via `RCP_API_KEY` from `.env`.
- **Model**: `Qwen/Qwen3-30B-A3B-Instruct-2507` only (no Apertus
  fallback). Best published multilingual instruction-following on the
  AIaaS menu at $0.05/$0.15 per 1M, MoE inference is fast at small
  batch sizes.
- **Position buckets**: 3 — head [0, total_chars/3),
  mid [total_chars/3, 2·total_chars/3), tail [2·total_chars/3,
  total_chars). Two queries per (doc, bucket) — one `question` plus
  one `topical-phrase` — so 400 docs × 3 buckets × 2 types = ~2400
  raw generations.
- **Query types**: every bucket emits both a `question` and a
  `topical-phrase` query (deterministic by construction; no RNG, no
  seed). Both types get equal coverage at every position so the
  downstream stratified Recall analysis can compare them directly.
  Summary-style queries are explicitly **out** (inflate Recall via
  lexical overlap; don't match real archive-user query distribution).
- **Languages**: fr / de — same as the corpus. The system prompt
  passes `lg` to the LLM and asks it to generate the query in the
  *same* language as the source. lb is dropped for the same reason
  step 1 dropped it (no eligible long-clean lb articles).
- **No cosine filter on v1.** Chroma's `(relevance ≥ 0.42, dedup ≤
  0.70)` filters are tuning knobs (Chroma blog: "selected through
  binary search with manual sampling validation"); we'd be picking
  thresholds before we have a baseline retention rate to tune
  against. v1 trusts the LLM + the verbatim-anchor verification (next
  bullet) and emits *every* surviving query so we can pick a
  threshold post-hoc. Listed as O7 below.
- **Verbatim-anchor verification.** Each LLM-emitted reference must
  appear as an exact substring of the source `ft`, *and* its first
  occurrence must fall inside the position bucket. References that
  don't appear, appear only outside the bucket, or are produced as
  modernised paraphrases are dropped. A query whose every reference
  is dropped is itself dropped. This is the only quality gate on v1.

## Mechanism

1. **Read corpus shard** via `s3io.iter_jsonl_bz2_path` into memory.
   Corpus is ≤400 records by design.
2. **Plan jobs.** For each record, enumerate the 3 buckets and the
   2 query types → 6 jobs per non-empty record. Empty buckets (very
   short docs) are filtered at planning time.
3. **For each job**, dispatched concurrently via `asyncio.gather`
   under an `asyncio.Semaphore(cfg.max_parallel)` for the AIaaS
   per-key cap (default 2; the published cap is 5 but real-world
   smoke runs throttled hard at 5, so 2 is the empirical sweet spot):
   1. Call the LangChain runnable returned by `make_llm(cfg)`. That
      runnable composes three primitives:
      - `ChatOpenAI(base_url=cfg.endpoint, …)` — the
        OpenAI-compatible chat client; `base_url` retargets it from
        api.openai.com to AIaaS / vLLM / any endpoint that speaks
        the OpenAI chat protocol.
      - `.with_structured_output(QueryOutput, method="json_mode")` —
        sets `response_format={"type": "json_object"}` on the API
        call AND parses the completion into a `QueryOutput`
        Pydantic model in one step. No hand-rolled fence stripping,
        no manual JSON validation.
      - `.with_retry(stop_after_attempt=cfg.retry_attempts,
        wait_exponential_jitter=True)` — retries on any exception
        (HTTP 429/503/504, network, parse error) with exponential
        backoff + jitter, courtesy of LangChain's tenacity-backed
        retry primitive.
   2. The system prompt:
      - Pins the output language to `lg`.
      - Asks for `{"query": str, "references": [str, ...]}` (the
        Pydantic schema is enforced separately by `json_mode`).
      - Forbids modernising / paraphrasing references.
      - Constrains the references to come from the bucket region
        (the prompt receives the bucket label and the bucket's text
        excerpt as a constraint hint, but the verbatim check below
        is the source of truth).
4. **Verify references.** For each reference string, find its first
   occurrence in `ft`. Drop the reference if (a) not found, or (b)
   `match_start` falls outside `[lo, hi)`. Drop the whole query if no
   reference survives.
5. **Write** surviving queries to `.jsonl.bz2` in deterministic order
   (`asyncio.gather` preserves input order; jobs are emitted in
   `(ci_id, bucket_idx, query_type)` order), then upload to S3.

## LLM client

`make_llm(cfg)` returns a LangChain `Runnable` whose `await ainvoke(messages)`
yields a parsed `QueryOutput` Pydantic instance per call. The
runnable composition (`ChatOpenAI → with_structured_output →
with_retry`) replaces three previously hand-rolled pieces:

- `urllib.request`-based HTTP client → handled by `ChatOpenAI`'s
  underlying `openai` SDK (which is a transitive dep of
  `langchain-openai`).
- Manual `parse_completion_content` (fence stripping + key checks +
  type checks) → handled by `with_structured_output(QueryOutput,
  method="json_mode")`. The Pydantic `QueryOutput` schema is the
  single source of truth for the LLM output shape; it raises a
  parse error if the completion can't be coerced.
- Manual retry loop with `await asyncio.sleep` and jitter math →
  `.with_retry(stop_after_attempt=N, wait_exponential_jitter=True)`,
  which delegates to tenacity under the hood.

Auth flows through `RCP_API_KEY` (loaded via `python-dotenv` at CLI
init, passed as `api_key=cfg.api_key` to `ChatOpenAI`). The
`base_url=cfg.endpoint` retarget is what makes the same
`ChatOpenAI` class work against AIaaS / vLLM / any OpenAI-compatible
backend. Imports are local to `make_llm` so unit tests that don't
exercise the LLM path don't pay the LangChain module-load cost.

## Prompts

Two system prompts in code (`SYSTEM_PROMPT_QUESTION`,
`SYSTEM_PROMPT_TOPICAL_PHRASE`), each parameterised by `lg` (`"fr"` /
`"de"`) and the position-bucket label. Both ask for the same JSON
schema. The prompts are kept short (under 300 words each) and
explicit about:

- Output language must match `lg`.
- The query must be answerable from the **bucket region only**.
- References are *verbatim* substrings of the source — no
  spelling-modernisation, no paraphrase, no truncation across
  hyphenation. Historical OCR artefacts (broken hyphens, archaic
  spellings) are preserved.
- The reply must be a single JSON object `{"query":..., "references":
  [...]}`. The user message contains the full source text plus the
  bucket text as a constraint excerpt.

We may iterate on these once the first run lands; the locked-in
contract is the **JSON shape** and the **bucket-constraint
semantics**.

## Cost & wallclock estimate

- 400 docs × 3 buckets × 2 types = **2400 generations**.
- Doc length p50 ~17–21 k chars ≈ ~5 k tokens input. Output cap 1500
  tokens (was 600 — bumped after a smoke run produced truncated JSON
  on long verbatim quotes); expected actual output ~400 tokens / call.
- Qwen3-30B-A3B prices: $0.05 / 1M input, $0.15 / 1M output.
- Input: 2400 × 5000 ≈ 12 M tokens × $0.05 = **$0.60**.
- Output (expected, ~400 tok/call): 2400 × 400 ≈ 1 M × $0.15 = **$0.15**.
- Output (worst case, hits 1500-tok cap): 2400 × 1500 ≈ 3.6 M × $0.15 = **$0.54**.
- **Total ≈ $0.8 (expected) / ≈ $1.1 (worst case)**.
- Wallclock at 2 parallel × ~3–5 s/generation ≈ **60–100 min**
  (the published cap is 5 but the AIaaS endpoint throttled hard at
  5 in practice; 2 is the empirical sweet spot — slower but doesn't
  spike retries).

These are estimates; the run log will record real numbers. The
`stats.json` sidecar is a deferred follow-up (O8 below).

## Rejected alternatives

- **Hand-rolled `urllib.request` HTTP client + custom JSON parser
  + custom retry loop.** Was the v1 implementation; replaced by
  LangChain's `ChatOpenAI` + `with_structured_output` + `with_retry`
  composition after a smoke run hit "Unterminated string …" parse
  errors that needed a more robust JSON pipeline. Trade-off:
  ~120 lines of code (HTTP client + parser + retry math) deleted in
  exchange for a `langchain-core` + `langchain-openai` dependency
  (which transitively pulls `openai`, `pydantic`, `httpx`, `tenacity`).
  Net win: fewer corner cases to maintain, structured output is
  schema-validated by Pydantic, retry semantics are well-tested
  upstream.
- **Direct `openai` SDK without LangChain.** Considered — does cover
  HTTP + retry + structured output (`response_format` + Pydantic),
  but lacks the `Runnable.with_retry` ergonomics and would still
  need a hand-rolled retry wrapper. LangChain's composition is
  thinner *at call sites* even if the dep tree is larger.
- **CaaS-vLLM fallback path.** Rejected at v1 to keep the script
  genuinely simple — one endpoint, one model, no swap logic. A
  CaaS variant on plain transformers (no vLLM) was added later as
  a *separate* script (`query_generate_local.py`); see the "CaaS
  variant" section below. Same prompts, same schema, same study
  YAML — picked alongside the AIaaS path, not as a fallback inside
  it.
- **Cosine filter at v1.** Rejected — Chroma's thresholds are
  per-corpus tuning knobs and we have no signal yet on what
  retention rate the LLM produces on OCR-noisy historical fr/de.
  Filtering with `Qwen/Qwen3-Embedding-8B` (the AIaaS embedding
  endpoint) is the natural follow-up if v1 surfaces too many noisy
  queries; deferred to O7.
- **Apertus-70B fallback A/B.** Rejected — user locked Qwen3 alone.
  Reduces blast-radius in v1; revisit only if Qwen3 hallucinates the
  wrong century or anachronistic vocabulary on a sample.
- **Quintile (5) buckets.** Rejected per scoping discussion — 3
  buckets surface a head-mid-tail asymmetry just as well at first
  pass and halve the generation cost. Promote to quintiles only if
  the 3-bucket result shows a non-monotonic edge effect that needs
  finer resolution. (Lost-in-the-Middle and RULER use 5; we
  intentionally coarsen for cost.)
- **Sentence-aligned passage subsampling** (pick a sentence range in
  the bucket, send only that to the LLM, ask for a query *about that
  passage*). Rejected because the Chroma protocol shows the **whole
  doc**: this is what makes references guaranteed-verbatim against
  the doc and lets the LLM choose the most queryable span in the
  bucket itself. Sending only a passage would force the LLM to query
  about whatever boilerplate landed in the slice.
- **Loading a tokenizer** to bucket by token offsets instead of char
  offsets. Rejected — adds a HuggingFace tokenizer download and ~1 s
  startup for a +/- 5–10 % bucketing precision that doesn't matter
  for the head-mid-tail distinction. Char offsets are first-class on
  the corpus (`len_chars` is in the manifest already).
- **Summary as a third query type.** Rejected — inflates Recall via
  lexical overlap with source (summaries paraphrase the doc) and
  doesn't match real Impresso archive-user behaviour (historians
  write topical phrases or short questions, not summaries).
- **LLM-judge "is this passage queryable" pre-filter.** ([Chroma
  2025 Generative Benchmarking](https://research.trychroma.com/generative-benchmarking).)
  Tempting for OCR-noisy historical text where ~20–40 % of random
  passages are mastheads / ads / weather; rejected on v1 because we
  pass the *whole doc* (not a random passage), so the LLM picks the
  queryable span itself. Promote to v2 only if v1 retention is poor
  *and* manual inspection shows boilerplate-heavy queries.
- **N>1 of the same query type per bucket.** v1 emits exactly one
  `question` and one `topical-phrase` per (doc, bucket). Generating
  N=2–3 of *each* type per bucket would densify the query set but
  doubles/triples cost without obviously buying eval power until O7
  (dedup) is calibrated. Listed as O10.
- **Random 80/20 (or any-ratio) question-vs-topical-phrase pick per
  bucket.** This was the design before the user's "why having a
  random decision here?" pushback — rejected because (a) it requires
  a seed + RNG state that is otherwise unused on this branch, (b) it
  gives uneven coverage per bucket × type cell, and (c) it makes
  cross-type comparison less clean (a `question` outperforming a
  `topical-phrase` could be confounded by which docs each happened
  to land on). Two queries per bucket, one of each type, is strictly
  more informative for the same per-doc cost factor of 2× vs the
  random scheme's 1×.
- **Native Impresso annotations as the relevance signal** (topic
  clusters, NER, text-reuse). Already rejected in CLAUDE.md scope;
  restated for closure — would make this branch incomparable to the
  published chunking-eval literature.

## Open items

- **O7 — cosine filter calibration.** Defer until v1 lands; if
  inspection shows >10 % of queries are off-topic or near-duplicates,
  add a `Qwen/Qwen3-Embedding-8B` filter at the AIaaS endpoint with
  Chroma's blog defaults (`relevance ≥ 0.42`, `dedup ≤ 0.70`) and
  tune from there. Filter and metric being in different vector
  spaces is acceptable for *quality control*, not for scoring.
- **O8 — `stats.json` sidecar.** Per-run summary uploaded next to
  `queries-v1.jsonl.bz2`: counts of generated / parsed / verified /
  dropped per `(lg, bucket, query_type)`, total cost, p50/p95
  latency. Useful for the eval step's stratification and for
  detecting drift across re-runs.
- **O9 — query-type ablation.** v1 emits both `question` and
  `topical-phrase` per bucket; the eval step can stratify by
  `query_type` to see which retrieves better at each chunk size. If
  one type is unambiguously stronger we may drop the other in v2.
- ~~O10 — N>1 per (bucket, query_type).~~ Closed by step 5: the
  `query_generation.queries_per_bucket` YAML knob (and matching
  `--queries-per-bucket` CLI flag) emits N samples per
  (doc, bucket, query_type) cell. `query_id` always carries a
  `__NN` sample-index suffix so consumers don't have to switch
  parsers based on the multiplicity. Default 1 preserves the
  6-queries-per-doc shape; raise to 2-3 for proportional LLM cost
  and statistical-power gain.
- **O11 — Qwen3 sanity-check on OCR-noisy historical fr/de.** A
  manual eyeball pass over the first ~30 queries per language to
  confirm the model isn't hallucinating wrong centuries / modernised
  toponyms / fabricated content. Trivial wallclock, blocking
  acceptance bar before the eval step trusts the output.

## CaaS variant — `query_generate_local.py`

A second entry point ships alongside the AIaaS script:
`impresso-research-query-generate-local`, backed by
`src/impresso_text_embedder/research/query_generate_local.py`. Same
study YAML, same prompts, same `QueryOutput` schema, same
verbatim-anchor verification, same output `.jsonl.bz2` shape — the
only difference is *where the model runs*. Downstream eval cannot
tell the two paths apart beyond the `gen_endpoint` field
(`"local:transformers"` vs the AIaaS URL). The AIaaS script is
unchanged.

**Why a second path.** The AIaaS endpoint has a per-key parallel
cap (empirically 2 was the sweet spot) and is shared infrastructure;
a Run:AI container with a single H100 can run the full study
self-contained, mirrors the production embed sweep's submission
shape, and isolates the experiment from AIaaS-side throttling drift.
Reverses the v1 "CaaS-vLLM fallback path — rejected by the user"
decision now that the user has asked for a CaaS option for this
specific reason.

**Inference recipe** (mirrors the production embedder's bf16
strategy from `.history/gpu-throughput/notes.md`):

- `transformers.AutoModelForCausalLM.from_pretrained(...,
  torch_dtype=torch.bfloat16, device_map="auto",
  attn_implementation=...)`. `torch_dtype` over the newer `dtype=`
  kwarg keeps us on the transformers 4.x line (the Dockerfile pins
  `<5`; Qwen3 needs `>=4.51`, which fits). `device_map="auto"`
  resolves to `cuda:0` under the single-GPU-per-container Run:AI
  pattern.
- `attn_implementation="sdpa"` is the **default**: built into
  PyTorch's `scaled_dot_product_attention`, dispatches transparently
  to Flash Attention v2 on bf16 + Ampere/Hopper, and works with the
  current `Dockerfile` out of the box (no `flash-attn` wheel
  required). `flash_attention_2` is available as a `--attention`
  opt-in once `flash-attn` is added to the image (already flagged in
  the Dockerfile comment as a follow-up gated on measurement).
  `eager` is the slow-baseline override for drift bisection.
- Chat template via `tokenizer.apply_chat_template(messages,
  add_generation_prompt=True)`; `model.generate(...,
  do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
  max_new_tokens=1500)` — the recommended sampling settings from
  the Qwen3-30B-A3B-Instruct-2507 model card. `temperature == 0`
  switches to greedy decoding (`do_sample=False`) for
  reproducibility checks. `tokenizer.padding_side="left"` so causal
  generation aligns even at `batch_size > 1` in a follow-up.
- `with torch.inference_mode()` around `.generate()` mirrors the
  production embedder's hot-path discipline.

**JSON parsing.** Pure transformers `.generate()` has no
`response_format={"type": "json_object"}` knob — that's a
vLLM/OpenAI-API surface. So `parse_query_output(text)` does what
LangChain's `with_structured_output(method="json_mode")` was doing
on the AIaaS side, in plain Python: strip an optional ```json
fence, locate the first `{ … }` object (handles the rare
leading-prose case), `orjson.loads`, then validate against the
shared `QueryOutput` Pydantic schema. Returns `None` on any
failure; `generate_one_local` retries up to `--max-retries`
(default 3) before tagging the job as `error_kind="api"`. No
constrained-decoding library (Outlines / Guidance / LM Format
Enforcer) — a Qwen3 instruction-tuned model emits well-formed JSON
reliably enough that the retry-on-parse-fail loop is sufficient at
v1.

**Concurrency**: synchronous, single inflight generation. The AIaaS
path runs `asyncio.gather` under a semaphore because the bottleneck
is per-key TCP parallelism; here the GPU is the bottleneck so a
single inflight request is the right shape. Batched generation
(left-padded prompts → one `model.generate` call across `B` jobs)
is a simple follow-up if wallclock matters, but variable-length
outputs make the gain modest at small B; left as a follow-up.

**Common helpers.** `query_generate_local.py` imports the prompt
builders (`build_system_prompt`, `build_user_message`), the schema
(`QueryOutput`, `Query`, `Reference`, `Job`, `JobResult`,
`GenerationStats`, `CorpusRecord`), the planner (`_plan_jobs`,
`bucket_ranges`), the verifier (`verify_references`), the I/O
(`read_corpus_shard`, `write_queries`), and the stats formatter
(`_format_stats`) directly from `query_generate.py`. Single source
of truth for the prompt + parsing contracts; an edit to either
script's prompts would land in both runs.

**Tests** (`tests/test_query_generate_local.py`, 19 cases, all
pass on CPU): `parse_query_output` (plain JSON / fences / leading
prose / garbage / truncated / schema violation), `generate_one_local`
with an injected `generate_fn` stub (happy path / parse-retry /
max-retries giveup / api error / out-of-bucket reference /
query_id format), `generate_queries_local` (six-queries-per-record
deterministic order, empty corpus), `config_from_args`
(study-YAML defaults, CLI override, no `RCP_API_KEY` requirement),
schema parity with the AIaaS path. No real model is ever loaded —
`load_local_llm` and `generate_completion` are out of scope here
since they need GPU + weights and belong with the
functional/GPU-correctness tests.

**Cost & wallclock estimate.** No per-token API charge. Wallclock
on a single H100, single inflight generation, ~3.3B active params
per token (Qwen3-30B-A3B is MoE): published 150–250 tok/s on
H100 single-GPU bf16 (community benchmarks). 2400 generations × ~400
output tokens ≈ ~960 k output tokens / 200 tok/s ≈ ~80 min
wallclock for a `study-v1`-shaped run. Roughly comparable to AIaaS
at 2-parallel; the win is "zero shared-infra coupling and
deterministic environment" rather than throughput.

**Rejected alternatives** (CaaS-side):

- **vLLM offline mode** (`vllm.LLM(...).generate(...)`). Tempting
  for `response_format=json` and continuous batching, but adds a
  large new dependency (50+ packages, transitive `xformers`/`triton`
  pins that may conflict with the NGC image) for a script that
  fits in ~400 lines on plain transformers. Promote only if (a)
  wallclock matters (continuous batching beats sync ~3-5×), or (b)
  schema-locked decoding becomes a quality-of-life requirement.
- **`outlines` constrained decoding.** Cleanest path to guaranteed
  schema-valid JSON, but adds a runtime dep, and Qwen3 emits
  well-formed JSON reliably enough that retry-on-parse-fail
  suffices. Promote if the parse-failure rate observed on a real
  run exceeds ~5%.
- **Default `attn_implementation="flash_attention_2"`.** Faster
  than SDPA on long contexts, but needs the `flash-attn` wheel
  (not currently in the Dockerfile per the comment in
  `Dockerfile:60-67`). Defaulted to `sdpa` so the script runs as-is
  on the production image; promoted to default once `flash-attn`
  ships in the image.
- **Multi-process / multi-GPU sharding inside one container.** Out
  of scope per `CLAUDE.md` → Non-goals — horizontal scaling is the
  Run:AI submission layer's job, not the script's. Submit two
  query-generate-local jobs over disjoint corpus shards if you need
  it.
- **Batched generation (`batch_size > 1`).** Not implemented at v1
  to match the user's "be as simple as possible" directive; the
  tokenizer is already configured `padding_side="left"` so adding
  batched generation later is a small mechanical change.

**Open items** (CaaS-specific):

- **OC1 — flash-attn wheel.** Add `flash-attn` to the Dockerfile
  alongside `xformers` and switch the local-script default to
  `flash_attention_2`. Trivial Dockerfile edit; gated on a real
  H100 wallclock measurement showing the SDPA→FA2 delta.
- **OC2 — batched generation.** Worth ~2-3× wallclock at modest
  complexity cost. Promote if the synchronous path puts a study
  run over the 2-hour Run:AI default queue cap.
- **OC3 — parse-failure rate sanity check.** Log the
  retry-attempts distribution per run; if >5% of jobs need
  retries, evaluate `outlines` (OC2 above).

## Reproducing

AIaaS path (default):

```
uv run impresso-research-query-generate \
  --config configs/research/study-v1.yaml \
  --max-parallel 2
```

`RCP_API_KEY` must be in `.env` (or exported).

CaaS path (single-GPU container):

```
runai submit --image <impresso-text-embedder-image> \
  --gpu 1 -- impresso-research-query-generate-local \
  --config configs/research/study-v1.yaml \
  --attention sdpa
```

Or for a local laptop smoke test (`--no-upload --limit 1` to
generate a single doc without round-tripping S3):

```
uv run impresso-research-query-generate-local \
  --config configs/research/study-v1.yaml \
  --no-upload --limit 1
```

## Upstream references

- Chroma Research, "Evaluating Chunking Strategies for Retrieval"
  (Jul 2024) — https://research.trychroma.com/evaluating-chunking
- `brandonstarxel/chunking_evaluation` GitHub repo (now
  `chroma-core/chunking_evaluation`) — prompt templates and
  threshold defaults.
- Chroma Research, "Generative Benchmarking" (Apr 2025) —
  https://research.trychroma.com/generative-benchmarking — chunk
  pre-filter rationale.
- Liu et al., "Lost in the Middle" (TACL 2024) —
  https://arxiv.org/abs/2307.03172 — head/mid/tail evaluation
  protocol.
- Hsieh et al., "RULER" (COLM 2024) —
  https://arxiv.org/abs/2404.06654 — depth-percentile NIAH
  stratification.
- Yang et al., "Qwen3 Technical Report" (May 2025) —
  https://arxiv.org/abs/2505.09388 — multilingual IF benchmarks
  including fr/de.
- Qwen3-30B-A3B-Instruct-2507 model card —
  https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507 —
  recommended sampling settings (`temperature=0.7`, `top_p=0.8`,
  `top_k=20`) and the minimum `transformers>=4.51.0` pin used by
  the CaaS variant.
- EPFL RCP AIaaS docs (internal portal,
  `https://inference.rcp.epfl.ch/v1` endpoint).
