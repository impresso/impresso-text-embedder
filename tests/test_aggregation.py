from __future__ import annotations

import numpy as np
import pytest

from impresso_text_embedder import aggregation
from impresso_text_embedder.aggregation.base import (
    _FACTORIES,
    AggregationStrategy,
    register_strategy,
)
from impresso_text_embedder.aggregation.mean import MeanPoolStrategy


class _Dummy(AggregationStrategy):
    def __init__(self, value=None):
        self.value = value

    def aggregate(self, vectors, weights=None):
        return np.zeros(vectors.shape[1], dtype=np.float32)


class TestRegistry:
    def test_mean_is_registered_by_default(self):
        assert "mean" in aggregation.available_strategies()

    def test_register_and_get_with_kwargs(self):
        register_strategy("dummy-test", _Dummy)
        try:
            got = aggregation.get_strategy("dummy-test", value=42)
            assert isinstance(got, _Dummy)
            assert got.value == 42
        finally:
            _FACTORIES.pop("dummy-test", None)

    def test_register_and_get_without_kwargs(self):
        register_strategy("dummy-noargs", _Dummy)
        try:
            got = aggregation.get_strategy("dummy-noargs")
            assert got.value is None
        finally:
            _FACTORIES.pop("dummy-noargs", None)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown aggregation strategy"):
            aggregation.get_strategy("does-not-exist")

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError):
            register_strategy("", _Dummy)

    def test_available_strategies_is_sorted(self):
        listed = aggregation.available_strategies()
        assert listed == sorted(listed)


class TestMeanPoolStrategy:
    def test_single_vector_passes_through_renormalised(self):
        strat = MeanPoolStrategy()
        v = np.array([[3.0, 4.0]], dtype=np.float32)  # norm 5
        out = strat.aggregate(v)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)

    def test_mean_of_unit_vectors_renormalises(self):
        strat = MeanPoolStrategy()
        # Two unit vectors separated by 90 degrees — mean has norm sqrt(2)/2 ≈ 0.707
        v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out = strat.aggregate(v)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)
        # Mean direction is (0.5, 0.5); renormalised to (1/sqrt(2), 1/sqrt(2))
        np.testing.assert_allclose(out, [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-6)

    def test_orthogonal_chunks_collapse_logs_warning(self, caplog):
        """Chunks pointing opposite directions ⇒ near-zero mean ⇒ WARNING."""
        strat = MeanPoolStrategy()
        v = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        with caplog.at_level("WARNING"):
            out = strat.aggregate(v)
        assert any("near zero" in r.message for r in caplog.records)
        # Output is returned un-normalised in the collapse branch
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_empty_raises(self):
        strat = MeanPoolStrategy()
        with pytest.raises(ValueError, match="empty"):
            strat.aggregate(np.zeros((0, 4), dtype=np.float32))

    def test_wrong_shape_raises(self):
        strat = MeanPoolStrategy()
        with pytest.raises(ValueError, match="2-D"):
            strat.aggregate(np.zeros(4, dtype=np.float32))

    def test_weights_argument_ignored(self):
        """Mean pool is deliberately unweighted; weights don't change the output."""
        strat = MeanPoolStrategy()
        v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out_unweighted = strat.aggregate(v, weights=None)
        out_weighted = strat.aggregate(v, weights=[100, 1])
        np.testing.assert_array_equal(out_unweighted, out_weighted)

    def test_output_is_float32(self):
        strat = MeanPoolStrategy()
        v = np.array([[1.0, 2.0]], dtype=np.float64)
        out = strat.aggregate(v)
        assert out.dtype == np.float32
