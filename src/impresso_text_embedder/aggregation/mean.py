"""Mean-pool aggregation: average chunk vectors and L2-renormalise.

Matches the PCW baseline in LongEmbed (arxiv 2404.12096) and the
Sentence-BERT convention. The only aggregation landed today;
length-weighted, max, and friends are deferred (see
``.history/long-doc-chunking/notes.md``).
"""

from __future__ import annotations

import logging

import numpy as np

from impresso_text_embedder.aggregation.base import AggregationStrategy

log = logging.getLogger(__name__)

# Below this pre-renormalisation norm, chunks are nearly orthogonal and the
# mean collapses to near-zero; the renormalisation then amplifies noise. We
# still return a unit vector in that case (so downstream cosine comparisons
# don't special-case ``NaN``), but log a warning so pathological documents
# are visible.
NEAR_ZERO_NORM_THRESHOLD = 1e-6


class MeanPoolStrategy(AggregationStrategy):
    """Unweighted mean over chunk vectors, followed by L2 renormalisation.

    ``weights`` is accepted for signature compatibility with the registry
    but deliberately ignored — length-weighted aggregation is a separate
    strategy and intentionally not baked in here.
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
        if k == 1:
            pooled = vectors[0]
        else:
            pooled = vectors.mean(axis=0)

        norm = float(np.linalg.norm(pooled))
        if norm < NEAR_ZERO_NORM_THRESHOLD:
            log.warning(
                "mean-pool norm collapsed to near zero (norm=%.3e, K=%d); "
                "returning un-normalised mean (vector is effectively noise)",
                norm,
                k,
            )
            return pooled.astype(np.float32, copy=False)
        return (pooled / norm).astype(np.float32, copy=False)
