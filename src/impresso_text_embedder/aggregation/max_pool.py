"""Max-pool aggregation: per-dimension max over chunk vectors, L2-renormalise.

Surfaces the most prominent feature per dimension across chunks. Less
stable than mean for retrieval (a single outlier chunk can dominate),
but useful as an A/B baseline against ``mean`` on multi-topic docs.
See ``.history/long-doc-chunking/notes.md`` § "Question 2 —
Aggregation strategies", option δ.
"""

from __future__ import annotations

import logging

import numpy as np

from impresso_text_embedder.aggregation.base import AggregationStrategy
from impresso_text_embedder.aggregation.mean import NEAR_ZERO_NORM_THRESHOLD

log = logging.getLogger(__name__)


class MaxPoolStrategy(AggregationStrategy):
    """Per-dimension max over chunk vectors, followed by L2 renormalisation.

    ``weights`` is accepted for signature compatibility with the registry
    but deliberately ignored — max pooling is inherently unweighted.
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
        k = vectors.shape[0]
        if k == 0:
            raise ValueError("cannot aggregate an empty set of vectors")
        pooled = vectors.max(axis=0)

        norm = float(np.linalg.norm(pooled))
        if norm < NEAR_ZERO_NORM_THRESHOLD:
            log.warning(
                "max-pool norm collapsed to near zero (norm=%.3e, K=%d); "
                "returning un-normalised max (vector is effectively noise)",
                norm,
                k,
            )
            return pooled.astype(np.float32, copy=False)
        return (pooled / norm).astype(np.float32, copy=False)
