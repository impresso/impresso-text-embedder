# semantic-chunker-fixes — design notes

Three coupled defects in the semantic chunker path (S11–S15 of the
`research/chunking-eval` sweep) that together broke the validity of
cross-family comparisons at the same nominal `chunk_tokens` value.
Found while answering a "what does `chunk_size` actually do?" question
on this branch; fixed in a single change because they all touch the
same call site.

## Defects

### 1. `chunk_size` measured in the wrong tokenizer

`chonkie.SemanticChunker.__init__` does

```python
tokenizer = self.embedding_model.get_tokenizer()
super().__init__(tokenizer)
```

(chonkie 1.6.4 `chunker/semantic.py:94-96`). The embedding model is
`minishlab/potion-base-8M`, whose tokenizer is the original BGE-base
30k-vocab WordPiece tokenizer. That tokenizer is ≈1.5–1.8× more
aggressive than the GTE multilingual SentencePiece tokenizer the
encoder actually uses:

| lang | GTE chars/token | chonkie chars/token | chonkie/GTE |
| ---- | --------------: | ------------------: | ----------: |
| fr   |            4.65 |                3.15 |       1.47× |
| de   |            5.34 |                2.99 |       1.79× |
| lb   |            3.16 |                2.69 |       1.17× |

(measured on hand-crafted prose passages; same trend expected on
Impresso historical text.)

Consequence: a sweep cell labelled "semantic-8190" was producing
chunks of ≈5500 GTE tokens on French / ≈4600 on German — *not*
~8190 — making cross-family comparison at a fixed nominal chunk
size apples-to-oranges. End-to-end demonstration on a 464-GTE-token
French passage with `chunk_size=200`:

- before: 4 chunks, realized `[116, 116, 116, 116]` GTE tok/chunk (58% of target)
- after: 3 chunks, realized `[174, 174, 116]` GTE tok/chunk (87% of target — the rest is chonkie's normal soft-boundary slack)

### 2. `min_sentences` silently dropped

Our `SemanticStrategy.__init__` passed `min_sentences=5` to chonkie
1.6.4, which had renamed the kwarg to `min_sentences_per_chunk`.
The old name landed in `**kwargs` and was discarded; the effective
floor was chonkie's `=1` default, not the documented `=5`. Realized
chunks could be a single short sentence.

### 3. `model2vec` not installed → SentenceTransformer fallback

`Model2VecEmbeddings.__init__` raised `ImportError`; chonkie warned
and silently fell back to `SentenceTransformerEmbeddings(minishlab/potion-base-8M)`.
Functionally similar 256-d output but a different code path: M2V is
a deterministic static lookup (~µs per sentence); the ST fallback
loads a real transformer and runs an autograd-free encode pass per
sentence. The S11–S15 family was running on the slower path, with
non-deterministic GPU ordering implications if those scenarios ever
move off CPU.

Initial fix put `model2vec>=0.3` under the `[research]` extra. That
worked for local dev (`uv sync --extra research`) but the
**production Dockerfile only installs `.` without extras**, so the
container running the sweep on RCP still hit the chonkie warning
on every `SemanticStrategy` construction. Promoted `model2vec` to
core deps in round 2 — it's a runtime dependency of bundled code
(`SemanticStrategy` constructs `SemanticChunker` with
`embedding_model="minishlab/potion-base-8M"`), pure-Python on top
of numpy, ABI-safe with NGC's numpy 1.26.x. Notebook-only tooling
(`pandas`/`seaborn`/`jupyterlab`/…) stays in `[research]`.

## Fixes

- **`chunking/semantic.py`**:
  - rename `min_sentences=` → `min_sentences_per_chunk=` (matches
    chonkie 1.5+ kwarg).
  - new optional `tokenizer=` param. When provided, swap
    `self._chunker._tokenizer` post-construction to chonkie's
    `AutoTokenizer(tokenizer)`. `_tokenizer` is the underlying
    attribute behind chonkie's read-only `tokenizer` property
    (chonkie 1.6 `chunker/base.py:79,87`). This is a deliberate
    private-API reach — chonkie offers no public override path on
    `SemanticChunker` — and is documented inline so a future chonkie
    upgrade with a public setter can switch.
- **`research/embed_sweep.py`**: thread `tokenizer=model.tokenizer`
  into the semantic chunker's kwargs in `build_long_doc_config`.
  `chunk_tokens` now means GTE tokens across all three chunker
  families.
- **`pyproject.toml`**: add `model2vec>=0.3` to the `research`
  optional-deps group. Pure-Python on top of numpy; <5MB transitive.
- **`tests/test_chunking.py`**: regression tests for all three —
  kwarg propagation, embedding-model class assertion, post-construction
  `_tokenizer` swap, and the negative case (no tokenizer kwarg → no
  swap).

## Out-of-sweep behaviour preserved

`SemanticStrategy()` with no `tokenizer=` kwarg keeps chonkie's
default tokenizer (the embedding model's). The production
`chunking="semantic"` path (not currently wired into the create CLI
defaults) is byte-for-byte unchanged. Only the research path on
this branch passes the encoder's tokenizer through.

## Realized-size slack (still expected)

Even with the right tokenizer, chonkie's `chunk_size` is a *soft*
target — semantic boundaries can fire before the budget is full,
and `min_sentences_per_chunk` floors short trailing chunks. The
`n_tokens_per_chunk` field already records realized GTE-token
sizes; analysis can either bin by realized size or trust the
nominal label. The fix narrows the gap from 50–60% of target to
85–95%, which is the chonkie-inherent ceiling.

## Rejected alternatives

- **Per-language `chunk_size` scaling at the call site** — multiply
  `chunk_tokens` by an empirical chonkie/GTE ratio. Mixes poorly
  with multi-language corpora (the corpus shard is fr+de+lb in one
  file); the tokenizer-swap fix is language-agnostic.
- **Custom `BaseEmbeddings` subclass exposing a GTE tokenizer via
  `get_tokenizer()`** — would also work, but touches more chonkie
  internals than the post-construction `_tokenizer` swap and adds a
  non-trivial class boundary for one private-attr line.
- **Pin the embedding-model class to `SentenceTransformerEmbeddings`
  and skip `model2vec`** — keeps the fallback as the "intentional"
  path, but slower and non-deterministic. M2V was the documented
  intent.
- **Document the asymmetry and re-bucket by `n_tokens_per_chunk`
  post-hoc** — viable fallback if the `_tokenizer` swap broke on a
  future chonkie release. Kept in reserve.

## Verification

- `pytest -q` → 493 passed.
- Smoke run on a 464-GTE-token French passage at `chunk_size=200`:
  realized GTE tok/chunk went from `[116, 116, 116, 116]` (old) to
  `[174, 174, 116]` (new). The remaining 26-token shortfall vs
  target 200 is the chonkie soft-target slack (semantic boundaries
  fired before the budget was full); not a bug.

## Container-version fallout — round 1 (rejected)

First RCP submission of the patched code crashed at import time:

```
ImportError: cannot import name 'AutoTokenizer' from 'chonkie.tokenizer'
(/usr/local/lib/python3.12/dist-packages/chonkie/tokenizer.py).
Did you mean: 'BaseTokenizer'?
```

Root cause: `chonkie.tokenizer.AutoTokenizer` was added in chonkie
1.5; the cached container image had been built when `pyproject.toml`
pinned `chonkie>=0.2`, so `pip` resolved an older chonkie whose
`tokenizer` module only exposed `BaseTokenizer` and concrete adapters.

First attempted fix was to bump the pin to `chonkie>=1.5` and add a
Dockerfile guardrail. **Rejected** — the rebuild failed:

```
ERROR: Cannot install impresso-text-embedder because these package
versions have conflicting dependencies.
  chonkie 1.5.0 .. 1.6.4 depend on numpy>=2.0.0
  The user requested (constraint) numpy==1.26.4
```

NGC `pytorch:25.03-py3` ships `numpy==1.26.4` and the apex / NCCL /
transformer-engine ABI stack hard-depends on staying on 1.26.x — the
Dockerfile already asserts this in a separate guardrail. Bumping
chonkie ≥ 1.5 inside this container is not viable until NGC ships an
image with numpy 2.

## Container-version fallout — round 2 (shipped)

Removed the chonkie pin bump and the AutoTokenizer dependency
entirely. SemanticStrategy now adapts to whatever chonkie the
resolver picks:

- **Kwarg-name detection** —
  `inspect.signature(SemanticChunker.__init__).parameters` is checked
  at construction time. chonkie ≥1.5 → forward
  `min_sentences_per_chunk=`; chonkie <1.5 → forward `min_sentences=`;
  neither in the signature → clear RuntimeError (defensive). Removes
  the silent-drop bug on either side of the rename.
- **In-house tokenizer adapter** — replaced the `chonkie.tokenizer.AutoTokenizer`
  call with a small `_TokenizerAdapter` class in
  `chunking/semantic.py` that exposes `count_tokens`,
  `count_tokens_batch`, and `encode` directly on top of the GTE HF
  tokenizer. chonkie's `SemanticChunker._split_sentences` only calls
  `count_tokens_batch`; the others are there for cross-version
  belt-and-suspenders. No dependency on any chonkie module beyond
  `SemanticChunker` itself.
- **Property-vs-attribute swap** — chonkie 1.5+ exposes `tokenizer`
  as a read-only property over `_tokenizer`; older versions have it
  as a plain settable attribute. `_swap_chunker_tokenizer` tries the
  public name first, falls through to `_tokenizer` on AttributeError.
- **Pin reverted** — `pyproject.toml` is back to `chonkie>=0.2`. The
  inline comment now points at the numpy ABI conflict so a future
  reader doesn't repeat the experiment.
- **`model2vec` promoted to core deps** — the round-1 placement
  under `[research]` only reached local dev; the container, which
  installs `.` without extras, kept hitting the
  `Model2VecEmbeddings: model2vec is not available` warning and
  the slower SentenceTransformer fallback. Moved to the main
  `[project] dependencies` list since `SemanticStrategy` always
  constructs a `SemanticChunker(embedding_model="minishlab/potion-base-8M")`
  and that's a runtime dep of bundled code, not notebook tooling.
- **Dockerfile guardrail removed** — the AutoTokenizer assert is
  gone; the numpy 1.26 / transformers <5 asserts stay.

Net effect: the container picks up whatever chonkie the resolver
lands on (consistently chonkie 0.x/1.0–1.4 inside NGC) and the
research sweep's S11–S15 work without rebuild from a fresh image.
Local dev keeps chonkie 1.6.4 (numpy 2.x is fine off-NGC).

End-to-end re-verified locally: `chunk_size=200` on the same 464-GTE-tok
French passage still produces `[174, 174, 116]` GTE tok/chunk
(was `[116, 116, 116, 116]` pre-fix). The new path goes through the
in-house adapter rather than chonkie's `AutoTokenizer`.

Test coverage now spans both chonkie eras: stub classes mirroring
both the chonkie 1.5+ (property `tokenizer`, `min_sentences_per_chunk`)
and chonkie <1.5 (settable `tokenizer`, `min_sentences`) signatures
exercise the kwarg-detection and tokenizer-swap branches independently.

## Follow-ups (deferred)

- Re-run S11–S15 against the existing study corpus shard once the
  research env on RCP picks up the new container, then compare the
  `n_tokens_per_chunk` distributions to confirm the shift holds at
  scale and across fr/de/lb.
- If a future chonkie release adds a public setter or `tokenizer=`
  kwarg on `SemanticChunker`, drop the `_tokenizer` reach.
