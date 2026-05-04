"""First-chunk aggregation: return the leading chunk's vector unchanged.

Equivalent to truncation but cut at a clean chunk boundary instead of at
the tokenizer's default. Mainly an A/B baseline for measuring how much
signal sits in the lead. See ``.history/long-doc-chunking/notes.md``
§ "Question 2 — Aggregation strategies", option ε.
"""

from __future__ import annotations

import numpy as np

from impresso_text_embedder.aggregation.base import AggregationStrategy


class FirstChunkStrategy(AggregationStrategy):
    """Return ``vectors[0]`` unchanged.

    Each chunk vector is already L2-unit (the model's ``Normalize`` module
    is asserted at load time), so no further renormalisation is needed.
    ``weights`` is accepted but ignored.
    """

    def aggregate(
        self,
        vectors: np.ndarray,
        weights: list[int] | None = None,
    ) -> np.ndarray:
        if vectors.ndim != 2:
            raise ValueError(
                f"expected 2-D [K, D] array, got shape {vectors.shape!r}"
            )
        if vectors.shape[0] == 0:
            raise ValueError("cannot aggregate an empty set of vectors")
        return vectors[0].astype(np.float32, copy=False)
