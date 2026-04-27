from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from impresso_text_embedder import model as model_mod


class TestSelectDevice:
    def test_picks_cuda_when_available(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            assert model_mod.select_device() == "cuda"

    def test_picks_cpu_when_no_cuda(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert model_mod.select_device() == "cpu"


class _FakeNormalize:
    """Stand-in for ``sentence_transformers.models.Normalize`` in tests."""


def _patched_st_modules(fake_ctor: MagicMock) -> dict[str, MagicMock]:
    """Sys.modules patch dict that satisfies both the ``SentenceTransformer``
    import and the ``from sentence_transformers.models import Normalize`` lookup
    inside :func:`load_model`'s assertion."""
    fake_models_mod = MagicMock()
    fake_models_mod.Normalize = _FakeNormalize
    return {
        "sentence_transformers": MagicMock(SentenceTransformer=fake_ctor),
        "sentence_transformers.models": fake_models_mod,
    }


class TestLoadModel:
    def test_forwards_expected_args(self):
        fake_ctor = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__getitem__.return_value = _FakeNormalize()
        fake_ctor.return_value = fake_instance

        with (
            patch.dict("sys.modules", _patched_st_modules(fake_ctor)),
            patch.object(model_mod, "select_device", return_value="cpu"),
        ):
            got = model_mod.load_model(name="foo/bar", revision="abc")

        fake_ctor.assert_called_once_with(
            model_name_or_path="foo/bar",
            trust_remote_code=True,
            revision="abc",
            device="cpu",
        )
        fake_instance.eval.assert_called_once()
        assert got is fake_instance

    def test_honours_explicit_device(self):
        fake_ctor = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__getitem__.return_value = _FakeNormalize()
        fake_ctor.return_value = fake_instance
        with patch.dict("sys.modules", _patched_st_modules(fake_ctor)):
            model_mod.load_model(device="cuda")
        assert fake_ctor.call_args.kwargs["device"] == "cuda"

    def test_default_auto_detect_enables_both_on_cuda_with_xformers(self):
        """Default ``None`` auto-detects → both flags enabled when CUDA + xformers."""
        fake_ctor = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__getitem__.return_value = _FakeNormalize()
        fake_ctor.return_value = fake_instance

        with (
            patch.dict("sys.modules", _patched_st_modules(fake_ctor)),
            patch.object(model_mod, "select_device", return_value="cuda"),
            patch.object(model_mod, "has_xformers", return_value=True),
        ):
            model_mod.load_model(name="foo/bar", revision="abc")

        assert fake_ctor.call_args.kwargs["config_kwargs"] == {
            "unpad_inputs": True,
            "use_memory_efficient_attention": True,
        }

    def test_default_auto_detect_disables_both_on_cpu(self):
        """Default ``None`` on CPU silently disables both — preserves pre-toggle behaviour."""
        fake_ctor = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__getitem__.return_value = _FakeNormalize()
        fake_ctor.return_value = fake_instance

        with (
            patch.dict("sys.modules", _patched_st_modules(fake_ctor)),
            patch.object(model_mod, "select_device", return_value="cpu"),
        ):
            model_mod.load_model()

        assert "config_kwargs" not in fake_ctor.call_args.kwargs

    def test_disables_xformers_omits_attention_flag(self):
        fake_ctor = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__getitem__.return_value = _FakeNormalize()
        fake_ctor.return_value = fake_instance

        with (
            patch.dict("sys.modules", _patched_st_modules(fake_ctor)),
            patch.object(model_mod, "select_device", return_value="cuda"),
            patch.object(model_mod, "has_xformers", return_value=True),
        ):
            model_mod.load_model(use_xformers=False)

        assert fake_ctor.call_args.kwargs["config_kwargs"] == {"unpad_inputs": True}

    def test_disables_unpad_omits_unpad_flag(self):
        fake_ctor = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__getitem__.return_value = _FakeNormalize()
        fake_ctor.return_value = fake_instance

        with (
            patch.dict("sys.modules", _patched_st_modules(fake_ctor)),
            patch.object(model_mod, "select_device", return_value="cuda"),
            patch.object(model_mod, "has_xformers", return_value=True),
        ):
            model_mod.load_model(unpad_inputs=False)

        assert fake_ctor.call_args.kwargs["config_kwargs"] == {
            "use_memory_efficient_attention": True
        }

    def test_disables_both_omits_config_kwargs_entirely(self):
        fake_ctor = MagicMock()
        fake_instance = MagicMock()
        fake_instance.__getitem__.return_value = _FakeNormalize()
        fake_ctor.return_value = fake_instance

        with (
            patch.dict("sys.modules", _patched_st_modules(fake_ctor)),
            patch.object(model_mod, "select_device", return_value="cuda"),
            patch.object(model_mod, "has_xformers", return_value=True),
        ):
            model_mod.load_model(use_xformers=False, unpad_inputs=False)

        assert "config_kwargs" not in fake_ctor.call_args.kwargs

    def test_use_xformers_on_cpu_raises(self):
        with (
            patch.dict("sys.modules", _patched_st_modules(MagicMock())),
            patch.object(model_mod, "select_device", return_value="cpu"),
        ):
            with pytest.raises(RuntimeError, match="CUDA"):
                model_mod.load_model(use_xformers=True)

    def test_use_xformers_without_xformers_package_raises(self):
        with (
            patch.dict("sys.modules", _patched_st_modules(MagicMock())),
            patch.object(model_mod, "select_device", return_value="cuda"),
            patch.object(model_mod, "has_xformers", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="xformers"):
                model_mod.load_model(use_xformers=True)

    def test_raises_when_last_module_is_not_normalize(self):
        fake_ctor = MagicMock()
        fake_instance = MagicMock()

        class _NotNormalize:
            pass

        fake_instance.__getitem__.return_value = _NotNormalize()
        fake_ctor.return_value = fake_instance

        with (
            patch.dict("sys.modules", _patched_st_modules(fake_ctor)),
            patch.object(model_mod, "select_device", return_value="cpu"),
        ):
            with pytest.raises(RuntimeError, match="Normalize"):
                model_mod.load_model(name="foo/bar", revision="abc")


def _mock_model_on(device: str, output: np.ndarray) -> MagicMock:
    m = MagicMock()
    m.parameters.return_value = iter([MagicMock(device=torch.device(device))])
    m.encode.return_value = output
    return m


class TestEncodeTexts:
    def test_empty_input_returns_empty_array_no_encode(self):
        m = _mock_model_on("cpu", np.zeros((0, 0), dtype=np.float32))
        out = model_mod.encode_texts(m, [], batch_size=32)
        assert out.shape == (0, 0)
        m.encode.assert_not_called()

    def test_cpu_path_forwards_kwargs(self):
        expected = np.random.rand(2, 4).astype(np.float32)
        m = _mock_model_on("cpu", expected)
        out = model_mod.encode_texts(m, ["a", "b"], batch_size=7)
        np.testing.assert_array_equal(out, expected)
        m.encode.assert_called_once()
        kwargs = m.encode.call_args.kwargs
        assert kwargs["batch_size"] == 7
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["normalize_embeddings"] is False
        assert kwargs["show_progress_bar"] is False

    def test_cuda_path_enters_bf16_autocast(self):
        expected = np.ones((1, 3), dtype=np.float32)
        m = _mock_model_on("cuda", expected)

        class _Recorder:
            entered = False

            def __enter__(self_inner):
                _Recorder.entered = True
                return self_inner

            def __exit__(self_inner, *a):
                return False

        def fake_autocast(*, device_type, dtype):
            assert device_type == "cuda"
            assert dtype is torch.bfloat16
            return _Recorder()

        with patch.object(torch, "autocast", fake_autocast):
            out = model_mod.encode_texts(m, ["x"], batch_size=1)
        assert _Recorder.entered is True
        assert out.dtype == np.float32

    def test_cpu_path_does_not_enter_cuda_autocast(self):
        m = _mock_model_on("cpu", np.zeros((1, 2), dtype=np.float32))
        autocast_mock = MagicMock()
        with patch.object(torch, "autocast", autocast_mock):
            model_mod.encode_texts(m, ["x"], batch_size=1)
        autocast_mock.assert_not_called()

    def test_fp32_precision_skips_autocast_on_cuda(self):
        """`precision='fp32'` skips the autocast scope even on CUDA."""
        m = _mock_model_on("cuda", np.zeros((1, 2), dtype=np.float32))
        autocast_mock = MagicMock()
        with patch.object(torch, "autocast", autocast_mock):
            model_mod.encode_texts(m, ["x"], batch_size=1, precision="fp32")
        autocast_mock.assert_not_called()

    def test_casts_float64_output_to_float32(self):
        out64 = np.ones((1, 2), dtype=np.float64)
        m = _mock_model_on("cpu", out64)
        out = model_mod.encode_texts(m, ["x"], batch_size=1)
        assert out.dtype == np.float32


@pytest.mark.parametrize(
    "dev,expected",
    [("cpu", "cpu"), ("cuda", "cuda"), ("mps", "mps")],
)
def test_current_device_reads_first_param(dev, expected):
    m = MagicMock()
    m.parameters.return_value = iter([MagicMock(device=torch.device(dev))])
    assert model_mod._current_device(m) == expected


def test_current_device_handles_parameterless_model():
    m = MagicMock()
    m.parameters.return_value = iter([])
    assert model_mod._current_device(m) == "cpu"
