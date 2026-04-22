# Migration plan — impresso-text-embedder

Living step list. Statuses: `todo` / `wip` / `done` / `deferred`. Slug names are stable; order can shift. See `/Users/adrien/.claude/plans/graceful-wishing-lake.md` for the mechanism this follows, and `CLAUDE.md` at repo root for package intent.

| # | Slug | Status | Notes folder? |
|---|---|---|---|
| 1 | package-skeleton | done | no |
| 2 | io-layer | done | `.progress/io-layer/` |
| 3 | schema-text-rebuild | wip | no |
| 4 | model-encoder | todo | `.progress/gpu-throughput/` |
| 5 | chunking | todo | `.progress/chunking/` |
| 6 | create-cli | todo | light |
| 7 | validate-cli | todo | `.progress/validation-metric/` |
| 8 | e2e-docs | todo | no |

## Step details

### 1. package-skeleton
`pyproject.toml` (hatchling + uv), `src/impresso_text_embedder/` layout, Python ≥3.10, ruff config, `LICENSE` (AGPL-3.0-or-later reused from `main`), `tests/` with placeholder, short `README.md` linking to `CLAUDE.md`, `.gitignore` update, delete `.flake8`. Verify: `uv sync` resolves, `uv run pytest` green.

### 2. io-layer
Wrap `impresso_essentials.io.s3` for `.jsonl.bz2` streaming, provider/alias/year iteration, idempotent skip-if-exists. Dotenv loaded at CLI boundary only.

### 3. schema-text-rebuild
Port `rebuild_ft_from_offsets`, `rebuild_sentence_from_offsets` verbatim. Dataclass/pydantic types for the three output schemas from `main:lib/text_embedding_processor.py`.

### 4. model-encoder
SentenceTransformer load + A100 optimizations (bf16 autocast, optional flash-attn, tunable batch size). Measurement deferred to real-GPU session; unit tests assert API shape only. Populate `.progress/gpu-throughput/notes.md`.

### 5. chunking
Registry + chonkie semantic strategy (threshold 0.5, chunk_size 1024, min_sentences 5). Contract in `.progress/chunking/notes.md`.

### 6. create-cli
`impresso-embed-create --provider ...` orchestrator, 1:1 mapping, async prefetch + upload overlap. Flags listed in `CLAUDE.md`.

### 7. validate-cli
`impresso-embed-validate <path> [--target ...] [--tol ...]`. Metric + default tol decided and documented in `.progress/validation-metric/notes.md`.

### 8. e2e-docs
Tiny local fixture end-to-end; polish `README.md`; resolve or explicitly defer remaining "Things to decide" items in `CLAUDE.md`.
