from __future__ import annotations

from unittest.mock import MagicMock, patch

from impresso_text_embedder.cli import create as create_cli


def test_parser_requires_provider_and_buckets():
    parser = create_cli.build_parser()
    # Missing everything
    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args([])
    # Only provider — still missing buckets
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "SNL"])


def test_parser_accepts_minimum_args():
    args = create_cli.build_parser().parse_args(
        [
            "--provider",
            "SNL",
            "--input-bucket",
            "i",
            "--output-bucket",
            "o",
        ]
    )
    assert args.provider == "SNL"
    assert args.embedding_level == "text"
    assert args.batch_size == 64
    assert args.content_type == ["ar"]
    assert args.chunking_strategy == "semantic"


def test_build_pipeline_config_threads_encoder_fields():
    args = create_cli.build_parser().parse_args(
        [
            "--provider",
            "P",
            "--input-bucket",
            "i",
            "--output-bucket",
            "o",
            "--batch-size",
            "128",
            "--min-char-length",
            "123",
            "--include-text",
            "--normalize-embeddings",
            "--content-type",
            "ar",
            "page",
        ]
    )
    cfg = create_cli._build_pipeline_config(args)
    assert cfg.encoder.batch_size == 128
    assert cfg.encoder.min_char_length == 123
    assert cfg.encoder.include_text is True
    assert cfg.encoder.normalize_embeddings is True
    assert cfg.encoder.content_types == frozenset({"ar", "page"})


def test_main_dry_run_skips_model_load():
    with (
        patch.object(create_cli, "process_provider", return_value={"processed": 0, "skipped": 0, "files": []}) as pp,
        patch.object(create_cli, "_load_env"),
        patch("impresso_text_embedder.model.load_model", side_effect=AssertionError("must not load")) as lm,
    ):
        rc = create_cli.main(
            [
                "--provider",
                "P",
                "--input-bucket",
                "i",
                "--output-bucket",
                "o",
                "--dry-run",
            ]
        )
    assert rc == 0
    lm.assert_not_called()
    assert pp.call_args.kwargs["dry_run"] is True
    assert pp.call_args.kwargs["model"] is None


def test_main_non_dry_run_loads_model():
    fake_model = MagicMock()
    with (
        patch.object(create_cli, "process_provider", return_value={"processed": 1, "skipped": 0, "files": ["k"]}) as pp,
        patch.object(create_cli, "_load_env"),
        patch("impresso_text_embedder.model.load_model", return_value=fake_model) as lm,
    ):
        rc = create_cli.main(
            [
                "--provider",
                "P",
                "--input-bucket",
                "i",
                "--output-bucket",
                "o",
            ]
        )
    assert rc == 0
    lm.assert_called_once_with(name="Alibaba-NLP/gte-multilingual-base", revision=None)
    assert pp.call_args.kwargs["model"] is fake_model
