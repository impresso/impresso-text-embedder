# Chunking — registry contract

## What's landing in step 5

- A `ChunkingStrategy` base class returning a list of `Chunk(text, start)` objects.
- A module-level registry of **lazy factories** (`name → () -> Strategy`). Lazy because some strategies pull in heavy deps (e.g. chonkie pulls its own embedding model) that we don't want to load unless the user asked for that strategy.
- One concrete strategy: `semantic` — ports the exact chonkie config from `main:lib/text_embedding_processor.py` (`minishlab/potion-base-8M`, threshold `0.5`, `chunk_size=1024`, `min_sentences=5`).

## Registry contract (for future strategies)

```python
from impresso_text_embedder.chunking import (
    Chunk,
    ChunkingStrategy,
    register_strategy,
    get_strategy,
    available_strategies,
)

class MyStrategy(ChunkingStrategy):
    def chunk(self, text: str) -> list[Chunk]:
        ...

register_strategy("my-strategy", MyStrategy)   # factory = class itself
strat = get_strategy("my-strategy")            # builds on first call; result is NOT cached
```

- `register_strategy(name, factory)` — `factory` is any zero-arg callable that returns a strategy instance.
- `get_strategy(name)` — builds a fresh instance every call. Callers that want a single instance per run should cache it themselves (cheap for semantic; the internal chonkie model is loaded lazily by chonkie).
- `available_strategies()` — sorted list of names, for CLI `--chunking-strategy` choices generation.

## `Chunk` shape

```python
@dataclass
class Chunk:
    text: str
    start: int | None = None   # character offset in the source text, if the strategy knows it
```

- `start` is optional because not every strategy can produce it. For the output schema this maps to the `o` field on `ChunkItem`.
- Strategies do **not** produce end offsets. The `text` already captures the span.

## Why not make the registry cache instances

- Chonkie loads an embedding model inside `SemanticChunker.__init__`; rebuilding per call would be wasteful, **but** if we cache instances in the registry, tests that want a fresh mock per test become annoying, and cross-CLI-invocation state is irrelevant (the CLI builds one instance anyway).
- Callers own caching. Typical flow: the CLI asks for the strategy once at startup and passes the instance into the per-document loop.

## Out of scope

- Token-aware chunking (respecting model's 8192 max-seq). The `semantic` strategy's `chunk_size=1024` is *characters* per chonkie, not tokens — we inherit that from the old code. A future `token-window` strategy should sit alongside semantic, not replace it.
- Re-encoding chunks with a different model than the main embedder (chonkie uses its own lightweight model for sentence similarity; this is separate from the Alibaba-NLP embedder used by our encoder).
