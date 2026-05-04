"""Length-weighted mean aggregation: weight each chunk by its token count.

Rationale: a 100-token chunk shouldn't count the same as a 1800-token
chunk when forming a document-level summary. Weights are required —
without them the strategy degenerates into plain mean and the caller
should ask for ``mean`` directly. See
``.history/long-doc-chunking/notes.md`` § "Question 2 — Aggregation
strategies", option γ.
"""

from __future__ import annotations

import logging

import numpy as np

from impresso_text_embedder.aggregation.base import AggregationStrategy
from impresso_text_embedder.aggregation.mean import NEAR_ZERO_NORM_THRESHOLD

log = logging.getLogger(__name__)


class LengthWeightedMeanStrategy(AggregationStrategy):
    """Weighted mean ``Σ wᵢ vᵢ / Σ wᵢ`` followed by L2 renormalisation."""

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
        if weights is None:
            raise ValueError("length-weighted mean requires per-chunk weights")
        if len(weights) != k:
            raise ValueError(
                f"weights length {len(weights)} does not match K={k}"
            )
        w = np.asarray(weights, dtype=np.float64)
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("sum of weights must be positive")
        pooled = (w[:, None] * vectors).sum(axis=0) / total

        norm = float(np.linalg.norm(pooled))
        if norm < NEAR_ZERO_NORM_THRESHOLD:
            log.warning(
                "length-weighted mean norm collapsed to near zero "
                "(norm=%.3e, K=%d); returning un-normalised mean "
                "(vector is effectively noise)",
                norm,
                k,
            )
            return pooled.astype(np.float32, copy=False)
        return (pooled / norm).astype(np.float32, copy=False)
