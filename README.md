# impresso-text-embedder

Multilingual text embeddings for [Impresso](https://impresso-project.ch) content items, using
[`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
via `sentence-transformers`. Designed for GPU-bound batch processing of yearly
`.jsonl.bz2` shards on S3 (A100-class hardware).

See [`CLAUDE.md`](./CLAUDE.md) for the full intent (data contract, target hardware,
non-goals, open decisions) and [`.progress/plan.md`](./.progress/plan.md) for migration
status and per-step design notes.

## Install

```bash
uv sync --extra dev
```

Credentials go in a local `.env` at the repo root (`SE_ACCESS_KEY`, `SE_SECRET_KEY`,
`SE_HOST_URL`). The file is gitignored.

## Usage

### Create embeddings for a provider

Walks `s3://<input-bucket>/[<prefix>/]<provider>/<alias>/<alias>-<year>.jsonl.bz2`
and writes
`s3://<output-bucket>/embeddings/docs/<model-slug>/<provider>/<alias>/<alias>-<year>.jsonl.bz2`,
one output file per input file.

```bash
uv run impresso-embed-create \
  --provider SNL \
  --input-bucket <input-bucket> \
  --output-bucket <output-bucket> \
  --embedding-level text \
  --batch-size 64
```

Useful flags: `--embedding-level {text,sentence,chunk}`, `--model-revision`,
`--alias EXP GDL --year-min 1910 --year-max 1920`, `--force`, `--dry-run`,
`--min-char-length 400`.

`--help` lists everything.

#### Embedding level vs. chunking strategy

`--embedding-level` picks *what* gets embedded:

- `text` — one embedding per content item from the full reconstructed text.
  No explicit long-text handling: inputs longer than the model's
  `max_seq_length` (8192 tokens) are silently truncated by the tokenizer.
- `sentence` — one embedding per sentence, using the input's pre-existing
  `sents` field. No chunking involved.
- `chunk` — splits the text via a chunking strategy and emits one embedding
  per chunk.

`--chunking-strategy` is only consulted when `--embedding-level=chunk`. It
selects an entry from the chunking registry (`src/impresso_text_embedder/chunking/`).
Currently registered: `semantic` (default) — `chonkie.SemanticChunker` with
`minishlab/potion-base-8M`, threshold `0.5`, chunk size `1024`, min
sentences `5`. `sentence` and `text` are *not* chunking strategies and are
not in the registry.

### Validate an output file

Structural check (no reference):

```bash
uv run impresso-embed-validate s3://<out>/.../EXP-1912.jsonl.bz2
```

Compare against a reference file (cosine distance per matching record, default
tolerance `1e-4`):

```bash
uv run impresso-embed-validate \
  s3://<out>/.../EXP-1912.jsonl.bz2 \
  --target s3://<out>/golden/EXP-1912.jsonl.bz2 \
  --tol 1e-4
```

Exit code is `0` on pass, `1` on failure.

## Development

```bash
uv run pytest             # unit + end-to-end tests
uv run ruff check .
```

## Containerised runs (EPFL RCP / Run:AI)

A `Dockerfile` and `Makefile` ship with the package for production runs on
the EPFL RCP cluster.

```bash
cp .env.docker.example .env.docker     # LDAP UID/GID, registry, Run:AI project
make docker-login
make docker-build-push                 # build linux/amd64 → Harbor
make k8s-create-secrets                # S3 + Harbor pull secret
make runai-submit PROVIDER=BNL \
     INPUT_BUCKET=22-rebuilt-final \
     OUTPUT_BUCKET=42-processed-data-final
```

`make help` lists every target. Design notes:
[`.progress/docker-runai/notes.md`](./.progress/docker-runai/notes.md).

See `.progress/<slug>/notes.md` for design decisions on individual subsystems
(I/O streaming, GPU throughput, chunking, CLI, validation metric).

## License

AGPL-3.0-or-later. See [`LICENSE`](./LICENSE).
