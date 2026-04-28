from __future__ import annotations

from unittest.mock import MagicMock, patch

from impresso_text_embedder.cli import create as create_cli


def test_parser_requires_provider():
    parser = create_cli.build_parser()
    import pytest

    # --provider is the only required flag; buckets fall back to defaults.
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_bucket_defaults():
    """Buckets have defaults pointing at the Impresso S3 workspace.

    Confirms both the default values and that the flags still accept overrides.
    """
    default = create_cli.build_parser().parse_args(["--provider", "SNL"])
    assert default.input_bucket == "122-rebuilt-final"
    assert default.output_bucket == "140-processed-data-sandbox"

    overrides = create_cli.build_parser().parse_args(
        ["--provider", "SNL", "--input-bucket", "i", "--output-bucket", "o"]
    )
    assert overrides.input_bucket == "i"
    assert overrides.output_bucket == "o"


def test_parser_accepts_minimum_args():
    args = create_cli.build_parser().parse_args(["--provider", "SNL"])
    assert args.provider == "SNL"
    assert args.embedding_level == "text"
    # Parser default is None; main() resolves it from accel.detect_profile().
    assert args.batch_size is None
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
            "--content-type",
            "ar",
            "page",
        ]
    )
    cfg = create_cli._build_pipeline_config(args)
    assert cfg.encoder.batch_size == 128
    assert cfg.encoder.min_char_length == 123
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


def test_parser_accepts_limit():
    default = create_cli.build_parser().parse_args(
        ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o"]
    )
    assert default.limit is None

    with_limit = create_cli.build_parser().parse_args(
        ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o", "--limit", "3"]
    )
    assert with_limit.limit == 3


def test_main_threads_limit_through_to_process_provider():
    with (
        patch.object(create_cli, "process_provider", return_value={"processed": 0, "skipped": 0, "files": []}) as pp,
        patch.object(create_cli, "_load_env"),
    ):
        rc = create_cli.main(
            [
                "--provider",
                "P",
                "--input-bucket",
                "i",
                "--output-bucket",
                "o",
                "--limit",
                "2",
                "--dry-run",
            ]
        )
    assert rc == 0
    assert pp.call_args.kwargs["limit"] == 2


def test_main_rejects_negative_limit(capsys):
    import pytest

    with pytest.raises(SystemExit):
        create_cli.main(
            [
                "--provider",
                "P",
                "--input-bucket",
                "i",
                "--output-bucket",
                "o",
                "--limit",
                "-1",
                "--dry-run",
            ]
        )
    err = capsys.readouterr().err
    assert "--limit must be >= 0" in err


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
    lm.assert_called_once_with(
        name="Alibaba-NLP/gte-multilingual-base",
        revision="f7d567e",
        use_xformers=True,
        unpad_inputs=True,
    )
    assert pp.call_args.kwargs["model"] is fake_model


def test_parser_pins_model_revision_by_default():
    args = create_cli.build_parser().parse_args(
        ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o"]
    )
    assert args.model_name == "Alibaba-NLP/gte-multilingual-base"
    assert args.model_revision == "f7d567e"


def test_parser_long_doc_flags_default_and_override():
    default = create_cli.build_parser().parse_args(
        ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o"]
    )
    # Default is 'chunk' (long docs go through the detect/chunk/aggregate
    # path); 'truncate' is the opt-out for reproducing pre-step-16 outputs.
    # chunk_tokens default is None — auto-derived from the loaded model's
    # tokenizer at CLI-init time (model_max_length - num_special_tokens_to_add).
    assert default.long_doc_strategy == "chunk"
    assert default.long_doc_chunk_tokens is None
    assert default.long_doc_aggregation == "mean"

    overrides = create_cli.build_parser().parse_args(
        [
            "--provider",
            "P",
            "--input-bucket",
            "i",
            "--output-bucket",
            "o",
            "--long-doc-strategy",
            "truncate",
            "--long-doc-chunk-tokens",
            "2048",
            "--long-doc-aggregation",
            "mean",
        ]
    )
    assert overrides.long_doc_strategy == "truncate"
    assert overrides.long_doc_chunk_tokens == 2048
    assert overrides.long_doc_aggregation == "mean"


def test_resolve_chunk_tokens_precedence():
    """CLI explicit value > tokenizer-derived > hard-coded fallback."""

    class _Tok:
        model_max_length = 4096

        def num_special_tokens_to_add(self, pair: bool = False) -> int:
            return 2

    tok = _Tok()
    # Explicit CLI value wins regardless of tokenizer.
    assert create_cli._resolve_chunk_tokens(1234, tok) == 1234
    # None → derive from tokenizer: 4096 - 2 = 4094.
    assert create_cli._resolve_chunk_tokens(None, tok) == 4094
    # Tokenizer missing the attributes → hard-coded fallback.
    assert (
        create_cli._resolve_chunk_tokens(None, object())
        == create_cli._FALLBACK_CHUNK_TOKENS
    )


def test_parser_rejects_unknown_long_doc_aggregation():
    import pytest

    # Only 'mean' is implemented today; additional choices are added as they
    # ship. argparse should reject anything else.
    with pytest.raises(SystemExit):
        create_cli.build_parser().parse_args(
            [
                "--provider",
                "P",
                "--input-bucket",
                "i",
                "--output-bucket",
                "o",
                "--long-doc-aggregation",
                "max",
            ]
        )


def test_dry_run_does_not_build_long_doc_config():
    """_build_long_doc_config needs a model; dry-run has none.

    The CLI must skip the long-doc wiring entirely in dry-run mode, even
    when ``--long-doc-strategy=chunk`` is passed (no crash, no tokenizer
    access).
    """
    with (
        patch.object(
            create_cli,
            "process_provider",
            return_value={"processed": 0, "skipped": 0, "files": []},
        ),
        patch.object(create_cli, "_load_env"),
        patch.object(
            create_cli, "_build_long_doc_config", side_effect=AssertionError("must not call")
        ) as bldc,
    ):
        rc = create_cli.main(
            [
                "--provider",
                "P",
                "--input-bucket",
                "i",
                "--output-bucket",
                "o",
                "--long-doc-strategy",
                "chunk",
                "--dry-run",
            ]
        )
    assert rc == 0
    bldc.assert_not_called()


def test_long_doc_chunk_wires_encoder_config():
    """``--long-doc-strategy=chunk`` attaches a LongDocConfig to the encoder."""
    from impresso_text_embedder.embed import LongDocConfig

    fake_model = MagicMock()
    captured = {}

    def _capture(args, model):
        # Record the call and return a real LongDocConfig so the rest of
        # the CLI path works.
        captured["args"] = args
        captured["model"] = model
        return LongDocConfig(
            strategy="chunk",
            chunker=MagicMock(),
            aggregator=MagicMock(),
            model_max_tokens=8192,
            token_counter=lambda t: 0,
        )

    with (
        patch.object(
            create_cli,
            "process_provider",
            return_value={"processed": 1, "skipped": 0, "files": ["k"]},
        ) as pp,
        patch.object(create_cli, "_load_env"),
        patch(
            "impresso_text_embedder.model.load_model", return_value=fake_model
        ),
        patch.object(create_cli, "_build_long_doc_config", side_effect=_capture),
    ):
        rc = create_cli.main(
            [
                "--provider",
                "P",
                "--input-bucket",
                "i",
                "--output-bucket",
                "o",
                "--long-doc-strategy",
                "chunk",
                "--long-doc-chunk-tokens",
                "2048",
            ]
        )
    assert rc == 0
    # _build_long_doc_config received the parsed args and the loaded model.
    assert captured["model"] is fake_model
    assert captured["args"].long_doc_chunk_tokens == 2048
    # The wired encoder now carries a LongDocConfig.
    cfg = pp.call_args.kwargs["cfg"]
    assert cfg.encoder.long_doc is not None
    assert cfg.encoder.long_doc.strategy == "chunk"


def test_parser_ablation_flags_defaults():
    """Defaults preserve the historical fast path: bf16 + xformers + unpad."""
    args = create_cli.build_parser().parse_args(
        ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o"]
    )
    assert args.precision == "bf16"
    assert args.attention == "xformers"
    assert args.unpad_inputs is True


def test_parser_ablation_flags_overrides():
    """`--precision`, `--attention`, `--no-unpad-inputs` flip the toggles."""
    args = create_cli.build_parser().parse_args(
        [
            "--provider",
            "P",
            "--input-bucket",
            "i",
            "--output-bucket",
            "o",
            "--precision",
            "fp32",
            "--attention",
            "eager",
            "--no-unpad-inputs",
        ]
    )
    assert args.precision == "fp32"
    assert args.attention == "eager"
    assert args.unpad_inputs is False


def test_build_pipeline_config_threads_precision():
    """The `--precision` flag lands on the EncoderConfig."""
    args = create_cli.build_parser().parse_args(
        [
            "--provider",
            "P",
            "--input-bucket",
            "i",
            "--output-bucket",
            "o",
            "--precision",
            "fp32",
        ]
    )
    cfg = create_cli._build_pipeline_config(args)
    assert cfg.encoder.precision == "fp32"


def test_main_threads_attention_and_unpad_into_load_model():
    """The CLI passes resolved use_xformers/unpad_inputs as kwargs to load_model."""
    fake_model = MagicMock()
    with (
        patch.object(
            create_cli,
            "process_provider",
            return_value={"processed": 1, "skipped": 0, "files": ["k"]},
        ),
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
                "--attention",
                "eager",
                "--no-unpad-inputs",
                "--long-doc-strategy",
                "truncate",
            ]
        )
    assert rc == 0
    kwargs = lm.call_args.kwargs
    assert kwargs["use_xformers"] is False
    assert kwargs["unpad_inputs"] is False


# --- step 18: multi-gpu file-list sharding -----------------------------------


def test_parser_shard_flags_default_to_no_op():
    """Defaults `--shard-index 0 --num-shards 1` reproduce single-job behaviour."""
    args = create_cli.build_parser().parse_args(
        ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o"]
    )
    assert args.shard_index == 0
    assert args.num_shards == 1


def test_parser_shard_flags_accept_overrides():
    args = create_cli.build_parser().parse_args(
        [
            "--provider", "P", "--input-bucket", "i", "--output-bucket", "o",
            "--shard-index", "2", "--num-shards", "4",
        ]
    )
    assert args.shard_index == 2
    assert args.num_shards == 4


def test_build_pipeline_config_threads_shard_fields():
    args = create_cli.build_parser().parse_args(
        [
            "--provider", "P", "--input-bucket", "i", "--output-bucket", "o",
            "--shard-index", "1", "--num-shards", "3",
        ]
    )
    cfg = create_cli._build_pipeline_config(args)
    assert cfg.shard_index == 1
    assert cfg.num_shards == 3


def test_main_rejects_zero_num_shards(capsys):
    import pytest
    with pytest.raises(SystemExit):
        create_cli.main(
            ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o",
             "--num-shards", "0", "--dry-run"]
        )
    assert "--num-shards must be >= 1" in capsys.readouterr().err


def test_main_rejects_shard_index_out_of_range(capsys):
    import pytest
    with pytest.raises(SystemExit):
        create_cli.main(
            ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o",
             "--shard-index", "4", "--num-shards", "4", "--dry-run"]
        )
    err = capsys.readouterr().err
    assert "--shard-index must be in [0, --num-shards)" in err


def test_main_rejects_shard_index_alone(capsys):
    """`--shard-index 2` without `--num-shards` is caught by the bounds check
    (2 not in [0, 1)) — same end result, clear error."""
    import pytest
    with pytest.raises(SystemExit):
        create_cli.main(
            ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o",
             "--shard-index", "2", "--dry-run"]
        )
    assert "--shard-index" in capsys.readouterr().err


def test_main_rejects_num_shards_alone(capsys):
    """`--num-shards 4` without `--shard-index` is the silent foot-gun guard:
    bounds-check passes (0 ∈ [0, 4)) but both-or-neither rule fires."""
    import pytest
    with pytest.raises(SystemExit):
        create_cli.main(
            ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o",
             "--num-shards", "4", "--dry-run"]
        )
    err = capsys.readouterr().err
    assert "must be set together" in err


def test_main_threads_shard_fields_into_configure_logging():
    """The CLI passes shard_index/num_shards into configure_logging so the
    log filename gets the per-shard suffix."""
    with (
        patch.object(create_cli, "process_provider", return_value={"processed": 0, "skipped": 0, "files": []}),
        patch.object(create_cli, "_load_env"),
        patch.object(create_cli, "configure_logging") as cfg_log,
    ):
        rc = create_cli.main(
            ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o",
             "--shard-index", "2", "--num-shards", "4", "--dry-run"]
        )
    assert rc == 0
    assert cfg_log.call_args.kwargs["shard_index"] == 2
    assert cfg_log.call_args.kwargs["num_shards"] == 4


def test_parser_log_flags_default_and_override(tmp_path):
    # Defaults: log-dir=None (resolve-time default), log-level-file=INFO.
    default = create_cli.build_parser().parse_args(
        ["--provider", "P", "--input-bucket", "i", "--output-bucket", "o"]
    )
    assert default.log_dir is None
    assert default.log_level_file == "INFO"

    overrides = create_cli.build_parser().parse_args(
        [
            "--provider",
            "P",
            "--input-bucket",
            "i",
            "--output-bucket",
            "o",
            "--log-dir",
            str(tmp_path),
            "--log-level-file",
            "DEBUG",
        ]
    )
    assert overrides.log_dir == tmp_path
    assert overrides.log_level_file == "DEBUG"


