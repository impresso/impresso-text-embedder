# Migration plan — impresso-text-embedder

Living step list. Statuses: `todo` / `wip` / `done` / `deferred`. Slug names are stable; order can shift. See `/Users/adrien/.claude/plans/graceful-wishing-lake.md` for the mechanism this follows, and `CLAUDE.md` at repo root for package intent.

| # | Slug | Status | Notes folder? |
|---|---|---|---|
| 1 | package-skeleton | done | no |
| 2 | io-layer | done | `.progress/io-layer/` |
| 3 | schema-text-rebuild | done | no |
| 4 | model-encoder | done | `.progress/gpu-throughput/` |
| 5 | chunking | done | `.progress/chunking/` |
| 6 | create-cli | done | `.progress/create-cli/` |
| 7 | validate-cli | done | `.progress/validation-metric/` |
| 8 | e2e-docs | done | no |
| 9 | docker-runai | done | `.progress/docker-runai/` |
| 10 | reembed-on-change | done | `.progress/reembed-on-change/` |

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

### 9. docker-runai
Container image + Run:AI submission for EPFL RCP. Reference: `feat/docker` branch. Deliverables:
- `Dockerfile` based on `nvcr.io/nvidia/pytorch:25.03-py3`, LDAP-matched user (PVC ownership), `pip install .` from copied source, `ENTRYPOINT ["impresso-embed-create"]`.
- `.env.docker.example` (LDAP UID/GID, registry/project, Harbor robot creds).
- `Makefile` (slim — docker build/push, k8s secrets, runai submit + interactive debug). No data-processing logic; the CLI handles that.
- `.gitignore` entries for `.env.docker` and `config.local.mk`.
- `CLAUDE.md` section linking to the workflow.
Decisions and gotchas in `.progress/docker-runai/notes.md`.

### 10. reembed-on-change
Re-embed an output when its input has been re-uploaded on S3, without resurrecting the old stamp tree or taking on the full `impresso_essentials.versioning` manifest system yet. Mechanism: compare S3 `LastModified` of input vs existing output at skip-decision time. `--force` still overrides unconditionally.
- `io.py`: add `last_modified: datetime | None` to `InputKey`, populate from `list_objects_v2` (no extra HEAD). Add `head_last_modified(bucket, key) -> datetime | None` (None on 404).
- `pipeline.py`: rewrite the skip branch in `process_file` and its dry-run twin in `process_provider` to compare timestamps. Distinct log lines for "skip (up-to-date)" vs "reprocess (input newer)".
- Tests: cover {output missing, output newer, output older, --force overrides}. Update existing `object_exists` patches to the new helper where the skip path is exercised.
- `CLAUDE.md`: record the decision under "Decisions recorded"; note the known gap (versioning manifest still deferred).
Rationale, trade-offs, and the explicit choice against option 2/3/4 in `.progress/reembed-on-change/notes.md`.
