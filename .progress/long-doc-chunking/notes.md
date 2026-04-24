# Long-document handling at text level — discovery

Status: **partially shipped** (see "Implementation status" below for the
subset landed on `feat/migration-python-package` and everything left
deferred). The bulk of this doc captures the design space that was
evaluated; treat any strategy not called out as shipped as still
available for a future incremental step. Decisions that need real-GPU
or real-corpus measurement are marked **OPEN**; fill them in as they
land.

## Implementation status (as of 2026-04-23)

**Shipped in this pass** — scope deliberately narrow so the architecture
can be validated on one end-to-end path before layering more strategies:

- New `aggregation/` module mirroring the chunking registry pattern:
  - `aggregation/base.py`: `AggregationStrategy` ABC,
    `register_strategy(name, factory)`, `get_strategy(name, **kwargs)`,
    `available_strategies()`. Kwargs-capable from day one so future
    strategies (length-weighted, attention-weighted, …) can slot in
    without signature churn.
  - `aggregation/mean.py`: `MeanPoolStrategy` — option **β** from the
    catalogue. Averages chunk vectors, L2-renormalises. Logs a
    WARNING when the pre-renormalisation norm collapses below
    `1e-6` (near-orthogonal chunks). `weights` argument accepted but
    ignored — length-weighted is a separate future strategy.
  - `aggregation/__init__.py`: registers `mean`.
- New `chunking/fixed_window.py`: `FixedWindowStrategy` — option **B**
  from the catalogue. Takes a tokenizer (HF-style, needs `encode` and
  `decode`) and a `max_tokens` ceiling. Tokenises the full text once,
  slices into contiguous non-overlapping windows of `max_tokens` ids,
  decodes each window back to a string. No sentence regex, no
  semantic clustering, no overlap — ~20 LOC. **Shipped as the
  default long-doc chunker** in the CLI. Optimisation level L1 (see
  below); L2/L3 deferred.
- New `chunking/token_budget.py`: `TokenBudgetStrategy` — option **D**
  from the catalogue. Takes a `token_counter: Callable[[str], int]`,
  a `max_tokens` budget, and an optional `sentence_splitter`. Greedy:
  packs whole sentences into ≤budget buckets; a single over-budget
  sentence is emitted as its own chunk (encoder will truncate — the
  documented escape hatch). Default splitter is a simple
  `.!?`-followed-by-whitespace regex (OCR-friendly, over-splits on
  "Mr." — acceptable since over-splitting just yields more-but-
  smaller chunks and chunk count doesn't affect output size).
  **Registered as `"token-budget"` but not wired as the CLI
  default**; reachable via the registry directly, and a one-line
  CLI flag addition away if we ever want A/B testing on multi-topic
  long docs.
- Chunking registry extended: `register_strategy` / `get_strategy`
  now pass kwargs through to the factory, so runtime deps (here,
  the tokenizer-derived token counter) can be injected without
  registry bloat. Backward compatible — existing `semantic` factory
  (zero-arg) still works. Token-budget registered as
  `"token-budget"`.
- `embed.py` refactor:
  - New `LongDocConfig` dataclass fielded on `EncoderConfig.long_doc`
    (defaults to `None` — no behaviour change when absent).
  - New `is_long_doc(text, cfg)` helper with a cheap char-estimate
    short-circuit before the real tokeniser call.
  - `_PendingText` now holds `texts: list[str]` (1 for short docs,
    K for chunked). `TextBatcher.add` runs the detect/chunk branch
    per-record; `TextBatcher.flush` encodes every pending text in
    one batched `encode_texts` call, then slices the vectors back
    per-record and applies the aggregator iff K > 1. Short-doc path
    is byte-identical to the pre-step-16 behaviour.
  - Per-file telemetry: new `LONG_DOC_CHUNKED` counter bumped once
    per long doc; surfaced through the existing filter-counter path
    to the per-file "done" INFO line.
- `cli/create.py`:
  - Three new flags: `--long-doc-strategy {truncate,chunk}`
    (default **`chunk`** — long docs are no longer silently
    truncated), `--long-doc-chunk-tokens N` (default **`None`**,
    meaning *auto-derive* at CLI-init time from the loaded model's
    tokenizer as
    `model_max_length - num_special_tokens_to_add(pair=False)`
    — 8190 for `gte-multilingual-base`; `_FALLBACK_CHUNK_TOKENS=8000`
    used only when the tokenizer doesn't advertise either attr),
    and `--long-doc-aggregation {mean}` (only `mean` is shipped;
    extra choices will appear here as they land). `truncate`
    remains the opt-out for reproducing pre-step-16 outputs
    exactly. CLI chunker is hardcoded to `fixed-window`; switching
    to `token-budget` is a one-line addition
    (`--long-doc-chunker`) if needed.
  - `_build_long_doc_config(args, model)` runs after model load,
    constructs the tokenizer-derived `token_counter`, builds the
    chunker and aggregator via the registries, and rebuilds the
    encoder config via `dataclasses.replace`.
- Tests:
  - `tests/test_aggregation.py`: registry happy/sad paths,
    `MeanPoolStrategy` (single vector, orthogonal-collapse warning,
    empty/wrong-shape rejection, weights ignored, fp32 output).
  - `tests/test_chunking_token_budget.py`: empty input, single
    short sentence, packing across multiple sentences, over-budget
    sentence, no-terminator text, custom splitter, registry entry.
  - `tests/test_embed.py::TestTextBatcherLongDoc`: short-doc bypass,
    long-doc chunk+aggregate, `strategy="truncate"` preserves
    legacy path, long + short in the same flush, chunker-returns-
    empty fallback-to-one-shot-with-warning.
  - `tests/test_embed.py::TestIsLongDoc`: detection logic
    (no-counter short-circuit, char-estimate short-circuit,
    triggered counter).
  - `tests/test_cli_create.py`: new-flag defaults, aggregation
    choices rejection, dry-run skips long-doc wiring, non-dry-run
    builds `LongDocConfig` and attaches it to encoder.
  - Full suite: 216/216 passing, ruff clean.
- **Default is `--long-doc-strategy=chunk`**: long docs flow through
  the detect → chunk → encode → aggregate path. This changes vectors
  for >8192-token inputs relative to pre-step-16 output; the old
  behaviour is recoverable with `--long-doc-strategy=truncate`.
  Goldens that include long docs need to be regenerated; short-doc
  outputs are byte-identical (the short path is unchanged).

**Not shipped — follow-up steps** (intentional scope gates, each is a
small incremental step):

- **Aggregation catalogue** — only `mean` is wired. Length-weighted
  mean (γ), max pool (δ), first-chunk baseline (ε), position-weighted
  (ζ), attention-weighted (η) are all designed in §"Question 2 —
  Aggregation strategies" and registered by adding a file under
  `aggregation/` and a `register_strategy` line in
  `aggregation/__init__.py`. CLI surface grows the `choices=` tuple
  on `--long-doc-aggregation`.
- **Chunking catalogue** — `fixed-window` (B, CLI default),
  `token-budget` (D, registered but not CLI-default), and the
  pre-existing `semantic` (J, chonkie-based, not token-aware) are
  registered. Stride overlap (C), paragraph-packer (E), recursive
  (F/I), chonkie Token/Sentence (G/H), Late Chunking (K — rejected
  here, see deep dive) are all designed in §"Question 1 — Chunking
  strategies" and would each be a new module under `chunking/`
  with a registry registration.
- **Optimisation of `fixed-window`** — shipped at level L1
  (straightforward: encode → slice → decode; `model.encode`
  re-tokenises). Per long doc this costs `2 + N` tokenisations +
  `N` decodes (N = chunk count) — invisible against GPU encode
  time for the few-percent of Impresso records that are actually
  long. L2 (hoist the detection tokenisation through to the
  chunker) saves 1 tokenise/doc; L3 (bypass
  `sentence-transformers.encode`, feed pre-tokenised id batches to
  the model forward pass) saves them all but requires reimplementing
  pooling + L2 norm + bf16 autocast + xformers unpadding. Both
  deferred pending profiling evidence that tokenizer overhead
  actually matters.
- **Sentence splitter upgrade** — the default regex is the MVP
  fallback. When `record["sents"]` is available (lingproc-enriched
  shards) we should prefer those boundaries. That requires either
  extending `ChunkingStrategy.chunk` to accept optional `sents`, or
  building the chunker per-record (closing over the sents). Deferred
  pending a concrete need.
- **`n_chunks` metadata on `TextRecord`** — gated on
  `embeddings-docs.schema.json`'s `additionalProperties` policy
  (OPEN-O4). For now the long-doc count is only visible through the
  per-file telemetry counter, not the per-record output.
- **Chunk-size calibration** — default `--long-doc-chunk-tokens=8000`
  is principled per the deep-dive (larger chunks preserve more
  cross-sentence attention in the forced-chunk regime for a
  CLS-pooled model) but a sweep over {512, 1024, 2048, 4096, 8000}
  on real Impresso long docs is still **OPEN-O2**.
- **`chars_per_token` per language** (OPEN-O3) — the fast
  char-estimate gate is a safe upper bound at 3.0 chars/token; a
  calibrated per-language value would skip more tokeniser calls on
  clearly-short inputs.
- **`--force` interaction** — the aggregated output for long docs
  differs from the legacy truncated output. When the default is
  flipped from `truncate` to `chunk`, a follow-up should document
  the expected cosine-distance range between old and new goldens
  (expect large distances on long docs by design, small on short).

## Two independent L2 normalizations — do not conflate

At text level with chunking on, there are **two** L2 normalization
steps applied at different pipeline stages on different inputs. They
are logically independent — neither subsumes or cancels the other.
Confusing them leads to wrong retrieval contracts.

### Step 1 — per-chunk normalization (inside the encoder)

For every encoded chunk:

```
tokens → Transformer → Pooling([CLS]) → Normalize → unit vector (‖v‖ = 1)
```

The model's `modules.json` for `gte-multilingual-base` is
`[Transformer, Pooling, Normalize]`, so every vector that comes out of
`model.encode(...)` is already L2-unit regardless of the user flag.
The `--normalize-embeddings` CLI flag (threaded through
`EncoderConfig.normalize_embeddings` into
`encode_texts(normalize=…)` → `model.encode(normalize_embeddings=…)`)
requests an **additional** L2 pass at the end of the encode pipeline.
On an already-unit vector this is the identity — a no-op — so the
flag is cosmetic for this model.

On a hypothetical model *without* a final `Normalize` module (a
CLS-pooled model that ships without the third module, or a
custom-built pipeline), the flag would matter: it would determine
whether chunks leave the encoder unit-norm or with their raw
post-pooling magnitudes. That distinction affects step 2 below.

### Step 2 — aggregate-level renormalization (inside MeanPoolStrategy)

Given K per-chunk vectors `v_1, …, v_K`:

```
pooled = mean(v_1, …, v_K)        # mean of K unit vectors; ‖pooled‖ ≤ 1
doc    = pooled / ‖pooled‖        # always unit-norm, even when K == 1
```

The mean of K unit vectors is **not** a unit vector in general
(equality holds only when all K point in the same direction). If we
skip the renormalization, the output vector's magnitude leaks
information about chunk coherence — a multi-topic doc's vector
becomes shorter than a single-topic doc's — which breaks the
`cosine = dot` invariant that `impresso-embed-validate` relies on
(`--tol 1e-4` cosine). Renormalization at this stage is required
whenever the caller wants a unit-norm document vector, which we
always do.

### Why the two are independent

1. **Different inputs.** Step 1 normalizes each chunk's output;
   step 2 renormalizes the *mean of K chunks*.
2. **Different necessity.** Step 1 is redundant for this model
   (built-in `Normalize` module) but would matter on a model that
   doesn't ship one. Step 2 is **always** necessary for unit-norm
   document output.
3. **Different mathematical meaning.**
   - Skipping step 1 on a model without built-in `Normalize` means
     the mean in step 2 implicitly becomes **norm-weighted** —
     bigger-norm chunks dominate the centroid direction. That's a
     distinct aggregation semantics from "direction-only averaging
     of unit chunks + renormalize."
   - `(a/‖a‖ + b/‖b‖) / 2` ≠ `((a + b)/2) / ‖(a + b)/2‖` in
     general; the two disagree whenever `‖a‖ ≠ ‖b‖`.
4. **Different pipeline stages.** Step 1 lives inside
   `sentence-transformers` (or inside the model's module list);
   step 2 lives in our `aggregation/mean.py`. They are compositional:
   you can change step 2 to a different aggregator (max, length-
   weighted, attention-weighted) without touching step 1, and
   vice-versa.

### Practical consequence for this repo

- For `gte-multilingual-base`: chunks are always unit-norm (step 1
  is done by the model). MeanPool's step 2 renormalization
  guarantees the aggregated document vector is also unit-norm.
- The `--normalize-embeddings` flag is therefore effectively
  orthogonal to the long-doc pipeline: flipping it neither
  introduces nor removes a normalization. We keep the default
  `False` to avoid giving the impression that it does something
  it doesn't.
- If we ever adopt a non-normalizing embedder, the same flag would
  start mattering — it'd determine whether chunks reach MeanPool
  pre-normalized or with raw magnitudes, which as shown above
  changes the aggregation semantics.
- Do not "simplify" by folding one step into the other. The
  aggregator must not assume chunks are unit-norm (future
  non-normalizing model would break); the encoder must not assume
  post-mean renormalization happens elsewhere (aggregators like
  `first-chunk` do not renormalize a mean — they short-circuit).

## Problem (pre-implementation framing; kept for context)

`--embedding-level text` produces **one D-dim vector per content item**
(D=768 for `gte-multilingual-base`, reducible via Matryoshka).
Today, `embed.py:147` hands the full document text to `model.encode(...)`,
which relies on the HF tokenizer's default `truncation=True`. Any content
past the model's `max_seq_length` (8192 tokens for `gte-multilingual-base`)
is **silently dropped** — no warning, no tally, no trace in the output.

For most Impresso items this is fine (news briefs, classifieds, short
articles are well below 8192). But the long tail — feuilletons, serialised
novels, parliamentary records, full-page political speeches — routinely
exceeds 8192 tokens. Their embedding today represents only the head of the
document. Retrieval over the tail is silently impossible.

We want a principled long-doc path that:

1. **Doesn't drop content by default.** Every token contributes to the final
   vector (within the quality limits of whatever aggregation we pick).
2. **Still emits one vector per content item** at `--embedding-level text`.
   The downstream Impresso pipeline consumes the document-embeddings schema
   (`{ci_id, embedding, size, …}`), which is inherently single-vector. A
   multi-vector path already exists at `--embedding-level chunk`; this
   discovery is specifically about the flat text level.
3. **Stays cheap.** 40+ years of Swiss/Luxembourg press is millions of
   items. A 2× encode cost per long doc is acceptable; 10× is not.

Two sub-questions drive the design:

1. **Chunking** — how to split the text into encoder-sized pieces.
2. **Aggregation** — how to fold K chunk embeddings back into one vector.

They are mostly independent; below we cover them separately and then
cross them in a single recommendation.

## Scope

- **In**: documents whose tokenised length exceeds the chunk ceiling we
  pick. `gte-multilingual-base` hard-caps at 8192; empirical evidence
  (below) suggests the *useful* ceiling is lower.
- **Out**:
  - Multi-vector outputs at text level. Covered by the existing
    `--embedding-level chunk`; changing the text-level schema to multi-vector
    would break Impresso's document-embeddings contract.
  - Late Chunking (Jina 2024). Discussed below for completeness — rejected
    for *our* long-doc path because it presupposes the full document fits
    in one forward pass, which is exactly what fails here. It's a
    potential future improvement for ≤8192-token docs at the `chunk` level,
    not a long-doc solution.
  - Cross-document aggregation, summarisation, or any LLM-assisted
    pre-processing. Pure encoder work only.

## Background: why the naive "just use 8192" path is wrong

Two independent signals say "don't just use the model's full context":

1. **BGE-M3 team's own recommendation.** BGE-M3 also supports 8192 tokens.
   Its maintainers explicitly say "splitting entire documents into multiple
   chunks often improves retrieval performance, and a chunk size of 512 is
   sufficient," because "mashing many topics and entities into a single
   vector representation doesn't produce good results"
   ([BAAI/bge-m3 HF discussion #59](https://huggingface.co/BAAI/bge-m3/discussions/59)).
   Empirical MRR@K on their tests collapsed when the whole 8192 window was
   used vs a chunked-and-aggregated alternative.

2. **LongEmbed benchmark (EMNLP 2024).** Short-context-trained embedders
   degrade past their natural training window even within their nominal
   `max_seq_length`. The Parallel Context Windows (PCW) baseline — split,
   encode each chunk, mean-pool — is the benchmark's standard approach to
   "handle a long doc with a short-context model"
   ([arxiv 2404.12096](https://arxiv.org/abs/2404.12096)). The paper shows
   position-encoding extension (RoPE/NTK/SelfExtend) works better than
   truncation but worse than explicit chunking for most real-world tasks.

The `gte-multilingual-base` model card is silent on the topic
([HF card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)) — it
advertises the 8192 capability but gives no recipe. The mGTE paper
([arxiv 2407.19669](https://arxiv.org/abs/2407.19669)) describes *training*
chunking, not *inference* chunking. We inherit the problem.

**Working assumption**: chunking at ~1024–2048 tokens beats both (a)
letting the tokenizer truncate and (b) loading the full 8192 in one shot
for multi-topic docs. We'll pick a concrete target ceiling at
implementation time and validate with Impresso data.

## Question 1 — Chunking strategies

All of these are token-aware (the model tokenizer decides boundary legality).
Character-based chunkers like the current `SemanticStrategy` (chonkie
`chunk_size=1024` *characters*) do not satisfy the 8192-token safety
constraint and are not in scope for this question.

### Catalogue

| Strategy | Description | Key param | External dep |
|---|---|---|---|
| **A. Truncate (baseline)** | Do nothing new. Tokenizer's `truncation=True` drops the tail. | — | none |
| **B. Fixed-token window, no overlap** | Split token stream into `[0:N], [N:2N], …`. | N=2048 | none (model.tokenizer) |
| **C. Fixed-token window, stride overlap** | `[0:N], [N-s:2N-s], …`. | N=2048, s=256 | none |
| **D. Sentence-aware token-budget packer** | Greedy: pack whole sentences into ≤N-token buckets, never split a sentence. | N=2048 | `sents` field already in input |
| **E. Paragraph-aware packer** | Same as D but units are paragraphs (`ft.split("\n\n")`). | N=2048 | none |
| **F. Recursive character splitter** | LangChain-style: paragraph→sentence→word fallbacks, token ceiling via model tokenizer. | N=2048, overlap=0 | `langchain-text-splitters` (small) |
| **G. chonkie `TokenChunker`** | Like B/C but via chonkie; uses tiktoken by default but accepts a custom tokenizer. | N=2048, overlap=0 | `chonkie` (already in deps) |
| **H. chonkie `SentenceChunker`** | Like D but via chonkie. | N=2048 | `chonkie` (already in deps) |
| **I. chonkie `RecursiveChunker`** | Like F but via chonkie. | N=2048 | `chonkie` (already in deps) |
| **J. chonkie `SemanticChunker` (token-aware mode)** | Current `SemanticStrategy` but with `chunk_size` interpreted as tokens and re-validated against model.tokenizer. | N=1024 tok | `chonkie` + extra embedding model (`minishlab/potion-base-8M`, already used) |
| **K. Late Chunking (Jina 2024)** | Full doc → encoder → per-token vectors → mean-pool inside chunk boundaries. | — | — |

### Analysis

**A (truncate)** is the current behaviour. It's the cheapest option and
the correct baseline for before/after A/B tests. Rejected as a production
default: silently drops content, unmeasurable recall impact.

**B (fixed token window)** is the dumbest thing that works. Two
pathologies: (1) splits inside words (xlm-roberta's SentencePiece at
least keeps tokens intact, so mid-word splits show up as suffixes, not
junk characters — but semantic boundaries are still broken), (2) no
signal that a chunk boundary falls in the middle of an argument. Okay
as a fallback inside D/E when a single sentence exceeds N.

**C (stride overlap)** is the sliding-window version. Overlap compensates
for boundary-induced context loss. Standard in long-doc QA where the
*location* of the answer matters. For a pure encoder producing one
vector per chunk that we'll then average, the overlap effectively
double-counts the overlap regions. Not obviously helpful here, and
**expensive** — a 25% overlap means 33% more encode work. Keep it as a
flag for experimentation, not a default.

**D (sentence-aware token-budget packer)** is the quality sweet spot.
Impresso records already carry a `sents` field from lingproc; when
present we get sentence boundaries for free. When absent (rebuilt-corpus
shards with only `ft`), we fall back to a lightweight sentence
splitter — for multilingual OCR'd text, `sents` is the right source of
truth anyway (language-specific splitters are unreliable on historical
spelling). The packer itself is ~30 lines: iterate sentences, maintain
a running token count, flush when adding the next sentence would
exceed N. On the rare pathological sentence > N tokens, fall back
locally to a B-style hard cut. **Strong candidate for the default.**

**E (paragraph-aware)** is a coarser D. For newspaper articles where
paragraphs are short and topical, it works well. For OCR'd text
paragraphs are less reliable than sentences (layout artifacts, column
wraps). Marginally simpler than D but less robust. Skip.

**F/I (recursive splitter)** is D's generalisation: try paragraph
boundaries first, fall back to sentence, then to word, then to char. On
clean text it's slightly better than D because it preserves paragraph
boundaries when it can. On our OCR'd corpus the paragraph signal is
noisy enough that the recursion rarely helps. Equivalent to D in
practice for our data, but adds complexity. LangChain's
`RecursiveCharacterTextSplitter.from_huggingface_tokenizer` exists but
pulls in langchain as a dependency for one helper; **not worth it**.
chonkie's `RecursiveChunker` is already in our deps — if we go this
route, use chonkie, not LangChain.

**G/H (chonkie Token / Sentence chunkers)** reuse the package we already
have. The open question is whether they accept `gte-multilingual-base`'s
HF tokenizer cleanly — chonkie's default is tiktoken (OpenAI), which
tokenises differently than xlm-roberta's SentencePiece. Mismatched
tokenisers mean the "≤2048 tokens" guarantee becomes "≤2048
tiktoken-estimated tokens, which is usually but not always ≤2048
xlm-roberta tokens". **OPEN** — verify at implementation time that
chonkie accepts `model.tokenizer` directly (recent versions support
arbitrary HF tokenizers via the `tokenizer` kwarg). If yes, H is
effectively the same as D with a library dep instead of 30 lines.

**J (semantic, token-aware)** is the current semantic strategy "fixed".
It clusters sentences by embedding similarity via
`minishlab/potion-base-8M` (~8M param model) and emits groups with
~topical coherence. It pays for an extra encoder pass on every
document. For the vast majority of Impresso items — short,
single-topic news articles — semantic chunking is overkill and the
per-doc overhead is a real throughput tax. For the long tail where a
single content item covers multiple topics (e.g. a full-page feature
with multiple unrelated briefs), semantic boundaries help. **Option,
not default.**

**K (Late Chunking, Jina 2024)**
([arxiv 2409.04701](https://arxiv.org/abs/2409.04701)) is structurally
different: run the full doc through the encoder *once* to get per-token
contextual vectors, then mean-pool those vectors inside each
post-hoc-defined chunk boundary. The killer advantage is that every
chunk's embedding incorporates context from the *entire* document, not
just its local window. Reported nDCG@10 gains: ~3.6% relative average
over naive chunking on long-doc datasets, up to 6.5 points on
NFCorpus ([Jina blog](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)).

**Why it doesn't solve our problem**: late chunking presupposes the
full document fits in one forward pass. For docs ≤8192 tokens it's a
quality upgrade for the `chunk` level (separate project). For docs
>8192 — exactly the case we're designing for — the full doc doesn't
fit, and you'd have to chunk *anyway* to get the per-token vectors,
which defeats the point. Late chunking is listed here because it comes
up in every long-doc RAG discussion; flag it as a **deferred
follow-up at `chunk` level only**.

### Ranking — chunking

| Strategy | Recall (expected) | Implementation LOC | Runtime overhead | Maintenance | Tier |
|---|---|---|---|---|---|
| A. Truncate | 2/5 — worst for long docs | 0 | 0 | 0 | reject-as-default, keep as opt-in baseline |
| B. Fixed token | 3/5 | ~20 | +1 tokenize pass per doc | low | **shipped as CLI default** — simpler beats clever for this model |
| C. Stride overlap | 3/5 — duplicates, muddies mean | ~25 | +33% encode per doc | low | experimental flag |
| D. Sentence-aware packer | **4/5** | ~40 | +1 tokenize + sentence re-count | low | shipped, registered as `"token-budget"`; opt-in (not the CLI default) |
| E. Paragraph packer | 3/5 | ~35 | same as D | low | skip |
| F/I. Recursive | 4/5 but complex | ~80 (or chonkie) | +1 pass | medium | option via chonkie |
| G/H. chonkie Token/Sentence | 4/5 | ~15 (lib) | library overhead | medium (chonkie churn) | option if D's hand-rolled path feels too DIY |
| J. Semantic (token-aware) | 4/5 on multi-topic, 3/5 on single-topic | ~20 (lib) | +1 potion-8M encode per doc | medium | option, not default |
| K. Late Chunking | N/A here — requires ≤8192 full-doc encode | — | — | — | deferred to `chunk` level |

**Shipped default (revised): B (`fixed-window`).** Tokenise the full
text once with `model.tokenizer`, slice into contiguous non-overlapping
windows of size `tokenizer.model_max_length -
tokenizer.num_special_tokens_to_add(pair=False)` (8190 for
`gte-multilingual-base`), decode each slice. No sentence regex, no
chonkie, no overlap. D (`token-budget`) was the original proposed
default; the reasoning was sentence-boundary preservation, but for a
CLS-pooled model where each chunk gets its own full forward pass the
preservation argument is weak and the regex surface area is pure
downside on OCR'd text. D stays registered as the opt-in
`"token-budget"` strategy — keep it for the day sentence-aware
packing proves empirically better on multi-topic docs.

## Question 2 — Aggregation strategies

K chunks → K per-chunk pooled vectors → one document vector (D=768
for gte-multilingual-base). Each per-chunk vector here is the model's
native CLS output on that chunk (see "Deep dive" below for why CLS
pooling matters); we are aggregating *model outputs*, not raw token
embeddings. The downstream Impresso text-level schema is fixed at one
vector per content item; multi-vector is covered by the `chunk` level
and out of scope here.

### Catalogue

| Method | Math | Unit norm? | Requires training? |
|---|---|---|---|
| α. No aggregation (multi-vector) | return K vectors | per-vector yes | no |
| β. Mean pool + L2 renormalise | `v = mean(v_i); v /= ‖v‖` | yes | no |
| γ. Length-weighted mean + renorm | `v = Σ w_i v_i / Σ w_i; v /= ‖v‖`, `w_i = tokens(chunk_i)` | yes | no |
| δ. Max pool | `v_j = max_i v_i,j` per dim; optional renorm | optional | no |
| ε. First-chunk only | `v = v_0` | yes | no |
| ζ. Position-weighted (lead bias) | `w_i = exp(-α·i)`, mean, renorm | yes | no |
| η. Attention-weighted (heuristic) | `α_i = softmax_i(v_i · μ)` where `μ = mean(v_i)`; `v = Σ α_i v_i`, renorm | yes | no |
| θ. Hierarchical re-encoder | Feed `[v_0, …, v_K]` through a small learned pooler | yes | **yes** |

### Analysis

**α. No aggregation.** Storing K vectors per doc blows up the index by
the average chunk count (probably 2–5× for Impresso). More importantly,
it **violates the text-level schema contract** — consumers (the
impresso-datalab, hnsw indices built downstream) expect one vector per
`ci_id`. Late-interaction retrieval (ColBERT, SPLADE) would gain here,
but that's a downstream re-architecting decision. **Reject for text
level.** This *is* already what the `chunk` level produces.

**β. Mean pool + renormalise.** The textbook choice. Sentence-BERT's
entire architecture is mean-pooling at a lower level; the same
invariants (dilution-resistance, stability, no training signal needed)
apply here. Used by the LongEmbed benchmark's PCW baseline
([arxiv 2404.12096](https://arxiv.org/abs/2404.12096)), by
Sentence-Transformers mean-of-chunks examples on the HF forum, and
implicitly by every "chunk and average" tutorial.

**Gotcha — "when to renormalise"**: mean-of-normalised-vectors is
**not** the same as normalise-of-mean-of-unnormalised. For cosine
retrieval we want the output to live on the unit sphere. The right
recipe, given that `encode_texts(..., normalize=True)` already L2-
normalises each chunk vector on return: average the normalised vectors,
then L2-renormalise the mean. Skipping the final renorm gives a
not-quite-unit vector (its norm reflects how parallel the chunks were),
which breaks the `cosine = dot` invariant used by the validate CLI.

**Stability note.** When chunks point in nearly-orthogonal directions
(very multi-topic document), `mean(v_i)` can collapse to near-zero and
the renormalisation amplifies whatever noise survives. In practice
Impresso articles are coherent enough that this is not a problem, but
we should log a warning if the pre-renorm norm is below, say, `0.1`.

**γ. Length-weighted mean.** Rationale: a 100-token chunk shouldn't
count the same as a 1800-token chunk when forming a "document-level"
summary. Intuition is sound. Empirical evidence is thin — most papers
compare unweighted mean vs other methods, not vs length-weighted. For
newspaper articles where lead paragraphs carry disproportionate
information, length-weighting actually *hurts* (lead is short, body
is long). **Option, not default.**

**δ. Max pool.** Emphasises the "most prominent" feature per dimension.
Works well when the document has one dominant topic that you want to
surface even if other chunks drown it out in the mean. Loses the
"average meaning" that cosine retrieval assumes. Generally produces
less stable vectors than mean pool
([Zilliz FAQ on pooling](https://zilliz.com/ai-faq/how-do-i-implement-embedding-pooling-strategies-mean-max-cls)).
For information retrieval specifically, mean pool is the standard.
**Reject as default.** Option for experimentation.

**ε. First-chunk only.** Equivalent to truncate (A) except we chose the
cut deliberately at a clean chunk boundary instead of at the tokenizer's
default. Useful as an A/B baseline showing "how much signal comes from
the lead." Not a real aggregation method. **Reject.**

**ζ. Position-weighted.** Newspaper-specific: the lede carries
disproportionate information. Weight early chunks higher. This is a
plausible prior for news but fragile for serialised literature and
political debates. Adding a hyperparameter (`α`) that's doc-type-
dependent is a maintenance smell. **Skip.**

**η. Attention-weighted (heuristic).** Re-weight chunks by how aligned
they are with the document's centroid. Intuitively attractive; in
practice for cosine retrieval it's a soft version of max pool and
shares max pool's instability. Needs at least one sweep to show a
gain over β, which I don't think is there based on the literature.
**Option, not default.**

**θ. Hierarchical re-encoder.** Requires training data and training
time. The quality ceiling is high (Longformer-style global attention
learned on top of chunk embeddings), but it's a different project.
**Reject.** Out of scope.

### Ranking — aggregation

| Method | Quality | Impl complexity | Runtime cost | Index cost | Training? | Tier |
|---|---|---|---|---|---|---|
| α. No agg (multi-vector) | high per-chunk, breaks schema | — | — | K× | no | reject (text level) |
| **β. Mean pool + renorm** | **4/5** | **1/5** | **1/5** | 1× | no | **ship default** |
| γ. Length-weighted mean | 4/5 (lateral move) | 2/5 | 1/5 | 1× | no | option |
| δ. Max pool | 3/5 for retrieval | 1/5 | 1/5 | 1× | no | option |
| ε. First-chunk | 2/5 | 1/5 | 1/5 | 1× | no | reject |
| ζ. Position-weighted | 3/5, fragile | 2/5 | 1/5 | 1× | no | skip |
| η. Attention-weighted | 3.5/5, unstable | 2/5 | 1/5 | 1× | no | option |
| θ. Hierarchical | 5/5 (ceiling), 0/5 (feasibility) | 5/5 | 5/5 | 1× | yes | reject |

**Working default: β (mean pool + L2 renormalise).** Ship as the
default aggregation. Expose `--long-doc-aggregation
{mean,max,length-weighted,first-chunk}` to enable A/B testing on a
real corpus without code changes. `mean` is the PCW baseline from
LongEmbed, is what every tutorial ships, and is numerically stable.

## Deep dive: for a doc ≤8192 tokens and single-vector output, is chunking ever better than one-shot?

This is a separate question from "what to do when the doc exceeds 8192",
and the two answers can diverge. The user specifically asked: for a text
that *fits* in context, which wins — (1) one-shot encode with the
model's native pooling, or (2) chunk, encode, mean-pool back to one
vector? Below is the analysis; the punchline is in the summary.

### Finding 1 — `gte-multilingual-base` uses **CLS pooling**, not mean pooling

Verified by fetching `1_Pooling/config.json` from the HF model repo
([raw file](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/raw/main/1_Pooling/config.json)):

```json
{
  "word_embedding_dimension": 768,
  "pooling_mode_cls_token": true,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}
```

The mGTE paper ([arxiv 2407.19669](https://arxiv.org/abs/2407.19669))
confirms: "we adopt the `[CLS]` token embedding as the text
representation." This single fact reshapes the comparison below.

Implication: the model's *trained* output lives in the CLS-token
subspace. Any aggregation recipe we apply post-hoc either stays in
that subspace (one-shot CLS, chunk-and-pool-the-CLS-vectors) or
leaves it (late chunking over mean-pooled token ranges — that
mean-pool produces vectors in a different geometry than the model's
training loss shaped).

### Finding 2 — late chunking's mathematical promise doesn't apply cleanly here

Late chunking ([Jina 2024](https://arxiv.org/abs/2409.04701))
assumes mean pooling: run the transformer on the full doc, get
per-token contextual embeddings, then mean-pool each chunk range.
For a **mean-pooled model**, late chunking with length-weighted
chunk aggregation is *mathematically identical* to one-shot
mean-pooling the whole doc, because

```
sum_k (|chunk_k|/N) * (sum_{i in chunk_k} h_i / |chunk_k|)
  = sum_k sum_{i in chunk_k} h_i / N
  = sum_{i=1..N} h_i / N
  = one-shot mean pool
```

(before the final L2 renormalisation; both are unit vectors after).
The value of late chunking is therefore **not** in single-vector
output — it's in producing multiple *contextualised* chunk vectors
for multi-vector retrieval. The Jina paper explicitly confirms this
framing: their "late chunking" evaluation is multi-vector chunk-level
retrieval, not single-vector-per-doc.

For `gte-multilingual-base`, late chunking is even weaker: because
the model uses CLS pooling, mean-pooling token-range outputs would
produce vectors in a subspace the model wasn't trained to populate.
The paper tested models like jina-embeddings-v2 (mean-pooled) and
nomic-embed-text (mean-pooled) — **not** a CLS-pooled bi-encoder.
Applying late chunking to `gte-multilingual-base` without retraining
is a research project, not a shipping decision.

### Finding 3 — Jina's own benchmarks show one-shot sometimes beats late chunking

Buried in Jina's blog post ([late-chunking-in-long-context-embedding-models](https://jina.ai/news/late-chunking-in-long-context-embedding-models/))
is a "No Chunking" column — what they call "No Chunking" is exactly
our one-shot single-vector-per-doc recipe:

| Dataset | No Chunking (single vector) | Naive Chunking (multi-vector) | Late Chunking (multi-vector) |
|---|---|---|---|
| NFCorpus | **30.40** | 23.46 | 29.98 |
| SciFact | (not tabulated) | 64.20 | 66.10 |
| FiQA2018 | (not tabulated) | 33.25 | 33.84 |
| Quora | (not tabulated) | 87.19 | 87.19 |

On NFCorpus, one-shot single-vector **beats** both naive and late
chunking. That's with a mean-pooled long-context model designed to
showcase late chunking's strengths — and one-shot still wins on that
dataset. On the others they don't publish the one-shot number, which
is itself suggestive.

### Finding 4 — BGE-M3's "chunk to 512" recommendation is a multi-vector regime finding

Re-reading the BGE-M3 team's guidance
([HF discussion #59](https://huggingface.co/BAAI/bge-m3/discussions/59))
more carefully: the comparison is *chunked and indexed as multiple
vectors* vs *one whole-document vector*. The chunked recipe wins
because the query can match any one of the K chunk vectors — not
because K chunk vectors averaged together is a better single-vector
representation. In the multi-vector regime, topic dilution is not a
problem (the query finds its matching chunk directly); in the
single-vector regime, chunking-then-averaging doesn't escape the
dilution, it just rearranges where the averaging happens.

This matters for our question because we're **locked into the
single-vector regime** at `--embedding-level text` (the Impresso
document-embeddings schema requires one vector per ci_id).

### Finding 5 — the model is natively trained at 8192 tokens

From the mGTE paper: a two-stage pre-training that first masks at
shorter lengths and then continues MLM at 8192 tokens. RoPE position
encoding. The model was designed for 8192 from the start, not extended
from a 512 base. The "model quality degrades near its max context"
argument that drives the BGE / E5 chunk-to-512 recommendation is
much weaker for mGTE.

### Finding 6 — information-theoretic argument

For a doc that fits in one forward pass:

- **One-shot encode** gives the CLS token attention to every other
  token in the document. Cross-chunk anaphora (e.g. "Berlin" in §1 →
  "the city" in §3) is resolved by the attention mechanism. The CLS
  vector is the model's trained representation of the whole
  document — optimised by the training loss for exactly this input.
- **Chunk-then-encode-then-pool-the-CLS-vectors** computes K
  independent CLS tokens, each attending only within its chunk. The
  "the city" mention in chunk 3's CLS has no knowledge of "Berlin" in
  chunk 1. The mean-of-K-CLS-vectors cannot recover this information
  — averaging doesn't create context that wasn't present.

So in the single-vector regime, chunking is strictly dropping
information and cannot outperform one-shot except under a narrow
set of conditions (topic-dominated docs where the mean-of-chunk-CLSes
happens to land closer to a topic-centroid than the full-doc CLS does
— no literature supports this, and our intuition is that it doesn't
hold for coherent news articles).

### Summary — direct answer to the user's question

| Option | Verdict | Why |
|---|---|---|
| **1. One-shot encode, no chunking (current short-doc path)** | ✅ **Winner** for ≤8192 tokens | Full attention, native CLS pooling, no information loss, aligned with model's training loss |
| **2a. Naive chunk + mean-pool CLS vectors** | ❌ Strictly worse | Loses cross-chunk attention; averaging doesn't recover the lost context |
| **2b. Semantic chunk + mean-pool CLS vectors** | ❌ Strictly worse + more expensive | Same info-theoretic issue as 2a, plus an extra `potion-8M` encode pass per doc |
| **2c. Late chunking + mean-pool over token ranges** | ❌ Doesn't apply | Mathematically equivalent to one-shot mean-pool *on a mean-pooled model*; gte-multilingual-base is CLS-pooled, so late chunking produces vectors in a different subspace than training |
| **Multi-vector (chunk level, indexed separately)** | ✅ Winner in a different regime | Out of scope — already what `--embedding-level chunk` produces |

**Decision: when the doc fits in the model's context, do NOT chunk.**
The existing single-vector text-level path (one-shot `model.encode`)
is the theoretically best and empirically competitive choice.
Chunking is worth doing *only* when the doc exceeds the model's
context — which is exactly the boundary step 16 was already proposing.

This resolves an apparent tension in the "naive long-doc default"
literature: the BGE-M3-style "chunk to 512" wisdom is a multi-vector
finding, and doesn't transfer to our single-vector schema. The step 16
working default already gets this right:

```
if token_count(doc) <= max_seq_length:
    one-shot encode          # this section confirms: this is optimal
else:
    chunk + mean-pool         # forced fallback, only because the doc doesn't fit
```

### Corollary — Late Chunking is not a promising follow-up at text level for this model

The earlier "deferred follow-up" note in the Chunking section suggested
Late Chunking could be a step-17 improvement at `--embedding-level
chunk`. Revised: that only makes sense if we switch to a mean-pooled
embedder, or if we validate empirically that mean-pooling
gte-multilingual-base's token outputs produces retrieval-competitive
vectors (non-obvious; requires a small benchmark). File this under
"if we ever change the embedder, reconsider"; do not treat it as a
pending improvement for the current model.

### Caveats and unresolved items

- **C1.** The Jina "No Chunking" numbers are on mean-pooled models,
  not CLS-pooled. For gte-multilingual-base specifically, we have no
  published benchmark comparing one-shot vs chunk+pool for
  single-vector output. The information-theoretic argument stands,
  but an empirical validation on a subset of Impresso data is cheap
  and worth doing once we have a golden query set.
- **C2.** For a multi-topic document that barely fits in 8192
  tokens (e.g. a full-page newspaper with 10 unrelated briefs
  concatenated), the one-shot CLS vector is still diluted — it's
  just that chunk+pool produces an equally-diluted vector via a
  different route. The real fix is multi-vector output, which is
  already available at `--embedding-level chunk`. If downstream
  Impresso consumers want better recall on multi-topic long docs,
  the answer is "use the chunk-level output," not "change the
  text-level aggregation."
- **C3.** If a future Impresso model switches to mean pooling (e.g.
  a bge-m3 or multilingual-e5 variant), late chunking becomes a
  legitimate option and this analysis should be redone.

### Sources cited in this section

- Pooling config for gte-multilingual-base: <https://huggingface.co/Alibaba-NLP/gte-multilingual-base/raw/main/1_Pooling/config.json>
- mGTE paper (CLS pooling, native 8192 training): <https://arxiv.org/abs/2407.19669>
- Late Chunking paper: <https://arxiv.org/abs/2409.04701>
- Jina blog with "No Chunking" numbers: <https://jina.ai/news/late-chunking-in-long-context-embedding-models/>
- BGE-M3 chunking discussion: <https://huggingface.co/BAAI/bge-m3/discussions/59>
- Chroma chunking evaluation (multi-vector focused): <https://www.trychroma.com/research/evaluating-chunking>

## Cross-cutting implementation notes

1. **Tokenisation cost.** Counting tokens with `model.tokenizer` is
   fast (Rust backend on the PreTrainedTokenizerFast) but not free.
   For docs that already fit in 8192, we shouldn't pay the tokenisation
   cost twice (once for our chunker, once inside `model.encode`). Two
   possible designs:
   - **Cheap token estimate.** Use `len(text) / avg_chars_per_token`
     as a first gate; only tokenise properly if it's close to the
     ceiling. xlm-roberta averages ~4 chars/token on Romance
     languages. **OPEN** — measure on Impresso data and pick a
     conservative estimate.
   - **Pay once.** Tokenise fully, split the token stream, pass
     pre-tokenised chunks into `model.encode` via its internal
     `features` API. Saves re-tokenisation in `encode`, complicates
     the code. Only worth it if profiling says tokenisation is a
     bottleneck.
   Ship the cheap-estimate approach; revisit on measurement.

2. **Schema and round-trip.** Output is unchanged — still a
   `TextRecord` with a single `embedding`. Optionally add a new
   optional field `n_chunks: int | None` so downstream consumers can
   tell which docs were long enough to trigger chunking. Useful for
   debugging and for later re-evaluation with a better strategy.
   **OPEN** — confirm the Impresso document-embeddings schema permits
   extra fields; if yes, add it; if no, log per-doc chunk counts to
   the file-level log only.

3. **Validation story.** The `--tol 1e-4` cosine validate contract is
   stable across chunk-then-aggregate, because the aggregation is
   deterministic. Generating a new golden is necessary one-shot the
   day we switch defaults; after that drift is measurable.

4. **Backwards compat.** Anyone currently using `--embedding-level
   text` against long docs has been getting truncated outputs. Switching
   the default aggregation changes their vectors. Options:
   - Default the new behaviour *on* and regenerate goldens.
     Preferred — the truncated behaviour is buggy, not a feature.
   - Gate behind `--long-doc-strategy {truncate,chunk-mean}` with
     default = `truncate`. Preserves exact reproducibility but ships
     a disabled feature. Doesn't justify the CLI complexity.
   - Gate behind a `--feature-flag` env var. Too subtle.
   **Ship with `chunk-mean` on by default**; document it in
   `CLAUDE.md` under "Decisions recorded" as a behaviour change.

5. **Interaction with existing `chunk` level.** The `chunk` level
   already produces multi-vector output. The new text-level long-doc
   path reuses the *chunking* logic (same strategy registry entry)
   but then aggregates. No duplicated code: both paths call the
   same `ChunkingStrategy.chunk(text)` method; the text-level path
   additionally calls `encode_texts` + aggregator. This argues for
   extending `ChunkingStrategy` to be *token-aware* (new method or
   new strategy) rather than adding a parallel hierarchy. See
   "Registry contract" in `.progress/chunking/notes.md` — the
   out-of-scope "token-window strategy should sit alongside
   semantic" note from step 5 is exactly the slot this fills.

## Recommendation

**Boundary**: chunking activates *only* when the doc exceeds the
model's context. For ≤8192-token docs we keep the current one-shot
path unchanged. See "Deep dive: for a doc ≤8192 tokens and
single-vector output, is chunking ever better than one-shot?" for
the information-theoretic and empirical rationale — briefly:
`gte-multilingual-base` uses CLS pooling, is natively trained at
8192, and chunking-plus-averaging in the single-vector regime
strictly drops the cross-chunk attention signal without any
compensating mechanism.

**Default wiring shipped (revised from the original proposal):**

1. `FixedWindowStrategy` (option **B**) is the CLI default long-doc
   chunker. Takes `model.tokenizer` and a `max_tokens` ceiling,
   slices `tokenizer.encode(text)` into contiguous windows,
   decodes each window. `TokenBudgetStrategy` (option D) is also
   shipped and registered as `"token-budget"` but is opt-in —
   reachable via the registry, not exposed via a CLI flag yet.
2. At `--embedding-level text`:
   - If the document's token count ≤ `model_max_tokens`, encode
     one-shot (unchanged fast path).
   - Otherwise, chunk with `fixed-window`, encode all chunks in a
     single `model.encode` call (batched alongside short-doc
     singletons), mean-pool + L2 renormalise, emit one
     `TextRecord`.
3. CLI flags:
   - `--long-doc-strategy {truncate,chunk}` (default `chunk`)
   - `--long-doc-chunk-tokens N` (default `None` — auto-derive
     from `tokenizer.model_max_length -
     tokenizer.num_special_tokens_to_add(pair=False)`; e.g. 8190
     for gte-multilingual-base)
   - `--long-doc-aggregation {mean}` (only mean is shipped;
     `max`, `length-weighted`, `first-chunk` are future additions)
4. Telemetry:
   - Per-file log: count of docs that triggered chunking, mean chunk
     count among those.
   - Per-doc DEBUG: `ci_id`, `n_chunks`, `chunk_tokens_histogram`.

**Defended against the runner-up.** The obvious alternative is
**length-weighted mean (γ)**. Rejected as default because (a)
literature shows no consistent win over unweighted mean, (b)
newspaper articles specifically have short-but-important lead
paragraphs that length-weighting penalises. Kept as an opt-in flag
for A/B experiments.

**Defended against the "just use 8192" path.** BGE-M3's own team
and empirical MRR results say single 8192-token embeddings are
worse than chunked-and-aggregated 512-token embeddings for
multi-topic docs. Our 2048 middle ground is a conservative starting
point; measurement on Impresso data should calibrate it.

## OPEN items to resolve during implementation

- **O1.** Does chonkie's `TokenChunker` / `SentenceChunker` accept
  `model.tokenizer` directly (HF PreTrainedTokenizerFast), or only
  tiktoken? If yes, prefer chonkie over hand-rolled (option H).
  If no, hand-roll (option D) — ~40 lines.
- **O2.** Optimal `--long-doc-chunk-tokens` default. BGE-M3's team
  says 512; mGTE trains at 8192; literature suggests the sweet spot
  for retrieval is corpus-dependent. Run a quick sweep (512, 1024,
  2048, 4096) on a handful of long Impresso items, compare cosine
  distance to human-judged relevance on a few queries. If we can't
  get human judgments, fall back to the BGE-M3 recommendation
  (512) or keep 2048 as a balanced guess.
- **O3.** Cheap-token-estimate ratio for the fast path. Measure
  `chars_per_token` on Impresso on each language (de/fr/it/lb).
  Pick a conservative bound; `>=ratio × max_seq_length` triggers
  the full tokenise.
- **O4.** `n_chunks` field in the `TextRecord`. Check the
  authoritative `embeddings-docs.schema.json` for additionalProperties
  policy. If restrictive, drop the field and rely on logs.
- **O5.** Late Chunking as a follow-up at `--embedding-level chunk`.
  Gains reported by Jina are real but only apply when the full doc
  fits in one forward pass. Worth a dedicated step once this one lands.

## Sources

- BGE-M3 chunking recommendation: <https://huggingface.co/BAAI/bge-m3/discussions/59>
- LongEmbed (EMNLP 2024): <https://arxiv.org/abs/2404.12096>
- mGTE paper (training, not inference chunking): <https://arxiv.org/abs/2407.19669>
- Late Chunking paper (Jina): <https://arxiv.org/abs/2409.04701>
- Late Chunking blog post with retrieval numbers: <https://jina.ai/news/late-chunking-in-long-context-embedding-models/>
- chonkie docs: <https://docs.chonkie.ai/common/open-source>
- LangChain `SentenceTransformersTokenTextSplitter`: <https://reference.langchain.com/python/langchain-text-splitters>
- Pooling strategies comparison: <https://zilliz.com/ai-faq/how-do-i-implement-embedding-pooling-strategies-mean-max-cls>
- Qdrant text chunking guide: <https://qdrant.tech/course/essentials/day-1/chunking-strategies/>
- Multi-dataset chunk-size analysis (2505.21700): <https://arxiv.org/html/2505.21700v2>
- "To chunk or not to chunk" with long-context models: <https://saeedesmaili.com/notes/to-chunk-or-not-to-chunk-with-the-long-context-single-embedding-models/>
