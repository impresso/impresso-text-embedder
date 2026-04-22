# impresso-text-embedder

Multilingual text embeddings for [Impresso](https://impresso-project.ch) content items, using
[`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
via `sentence-transformers`.

This branch (`feat/migration-python-package`) is an in-progress migration from the previous
Make/scripts layout on `main` to a `pyproject.toml`-based Python package. See
[`CLAUDE.md`](./CLAUDE.md) for the package intent, data contract, target hardware (A100), and
open decisions, and [`.progress/plan.md`](./.progress/plan.md) for migration status.

## Install (dev)

```bash
uv sync --extra dev
```

## CLIs

Not wired up yet — see `.progress/plan.md` steps 6 and 7. Intended surface:

```bash
uv run impresso-embed-create --provider <PROVIDER>
uv run impresso-embed-validate <path-to-jsonl.bz2> [--target <path>] [--tol <float>]
```

## License

AGPL-3.0-or-later. See [`LICENSE`](./LICENSE).
