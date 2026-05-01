from __future__ import annotations

import bz2
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import orjson
import pytest

from impresso_text_embedder import embed as em
from impresso_text_embedder.research import embed_sweep
from impresso_text_embedder.research.scenarios import Scenario


# ---------------------------------------------------------------------------
# Test fixtures — Scenario instances built directly (no registry needed for
# pure-function tests).
# ---------------------------------------------------------------------------


def _scenario_truncate() -> Scenario:
    return Scenario(
        id="S0",
        label="truncate-8192",
        chunker_name=None,
        chunk_tokens=None,
        aggregator_name=None,
    )


def _scenario_fixed_window(chunk_tokens: int, sid: str = "S1") -> Scenario:
    return Scenario(
        id=sid,
        label=f"fixed-window-{chunk_tokens}",
        chunker_name="fixed-window",
        chunk_tokens=chunk_tokens,
        aggregator_name="mean",
    )


def _scenario_token_budget(chunk_tokens: int, sid: str = "S7") -> Scenario:
    return Scenario(
        id=sid,
        label=f"token-budget-{chunk_tokens}",
        chunker_name="token-budget",
        chunk_tokens=chunk_tokens,
        aggregator_name="mean",
    )


def _study_v1_path() -> Path:
    return (Path(__file__).parent.parent / "configs/research/study-v1.yaml").resolve()


def _write_corpus(path: Path, records: list[dict]) -> None:
    with bz2.open(path, "wb") as fh:
        for r in records:
            fh.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))


def _corpus_record(
    *,
    ci_id: str,
    ft: str = "Hello world this is a long enough article body for the embed run.",
    lg: str = "fr",
    year: int = 1920,
    provider: str = "BNF",
    alias: str = "jdpl",
    len_chars: int = 20000,
    ocrqa: float = 0.95,
    tp: str = "ar",
) -> dict:
    return {
        "ci_id": ci_id,
        "lg": lg,
        "year": year,
        "provider": provider,
        "alias": alias,
        "len_chars": len_chars,
        "ocrqa": ocrqa,
        "tp": tp,
        "ft": ft,
        "sents": None,
    }


def _make_fake_model(dim: int = 4):
    """Fake SentenceTransformer with the attrs and tokenizer ``embed_sweep`` touches."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False, verbose=False: list(
        range(len(text))
    )
    # FixedWindowStrategy slices ids and decodes back to a string per chunk;
    # the new _compute_chunk_stats then re-tokenises the decoded chunk text,
    # so decode must return a real string rather than a MagicMock.
    tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: "x" * len(ids)
    model.tokenizer = tokenizer
    return model, dim


def _patch_encode(monkeypatch, dim: int = 4) -> None:
    def fake(_model, texts, **_):
        return np.asarray(
            [[float(i), float(len(t)), 0.0, 0.0] for i, t in enumerate(texts)],
            dtype=np.float32,
        )

    monkeypatch.setattr(em, "encode_texts", fake)


# ---------------------------------------------------------------------------
# build_long_doc_config
# ---------------------------------------------------------------------------


def test_build_long_doc_config_returns_none_for_truncate() -> None:
    assert embed_sweep.build_long_doc_config(_scenario_truncate(), MagicMock()) is None


def test_build_long_doc_config_fixed_window_uses_chunk_tokens_as_trigger() -> None:
    s = _scenario_fixed_window(512)
    model, _ = _make_fake_model()
    long_doc = embed_sweep.build_long_doc_config(s, model)
    assert long_doc is not None
    assert long_doc.is_active()
    assert long_doc.model_max_tokens == s.chunk_tokens == 512


def test_build_long_doc_config_token_budget_passes_token_counter() -> None:
    s = _scenario_token_budget(1024)
    model, _ = _make_fake_model()
    long_doc = embed_sweep.build_long_doc_config(s, model)
    assert long_doc is not None
    assert long_doc.is_active()
    assert long_doc.model_max_tokens == s.chunk_tokens == 1024


# ---------------------------------------------------------------------------
# _compute_chunk_stats
# ---------------------------------------------------------------------------


def test_compute_chunk_stats_truncate_records_total_tokens() -> None:
    """S0 has no chunker; each doc is one chunk and n_tokens == n_tokens_per_chunk[0]."""
    model, _ = _make_fake_model()
    counter = embed_sweep._build_token_counter(model)
    records = [
        {"ci_id": "a", "ft": "x" * 100},
        {"ci_id": "b", "ft": "y" * 1000},
        {"ci_id": "empty", "ft": ""},
    ]
    stats = embed_sweep._compute_chunk_stats(records, counter, long_doc=None)
    assert stats["a"] == embed_sweep.ChunkStats(
        n_chunks=1, n_tokens=100, n_tokens_per_chunk=[100]
    )
    assert stats["b"] == embed_sweep.ChunkStats(
        n_chunks=1, n_tokens=1000, n_tokens_per_chunk=[1000]
    )
    assert stats["empty"] == embed_sweep.ChunkStats.empty()


def test_compute_chunk_stats_chunked_path_records_per_chunk_tokens() -> None:
    s = _scenario_fixed_window(512)
    model, _ = _make_fake_model()
    long_doc = embed_sweep.build_long_doc_config(s, model)
    counter = embed_sweep._build_token_counter(model)
    # is_long_doc has a cheap len(text)/3.0 early-exit so text must clear
    # 512 * 3 = 1536 chars before the real tokenizer count is consulted.
    records = [
        {"ci_id": "short", "ft": "x" * 100},
        {"ci_id": "long", "ft": "x" * 2000},
    ]
    stats = embed_sweep._compute_chunk_stats(records, counter, long_doc=long_doc)
    # under threshold → one-shot, n_tokens == 100
    assert stats["short"] == embed_sweep.ChunkStats(
        n_chunks=1, n_tokens=100, n_tokens_per_chunk=[100]
    )
    # 2000 fake tokens / 512-id windows → 4 chunks of [512,512,512,464]
    long_stats = stats["long"]
    assert long_stats.n_chunks == 4
    assert long_stats.n_tokens == 2000
    assert long_stats.n_tokens_per_chunk == [512, 512, 512, 464]
    assert sum(long_stats.n_tokens_per_chunk) == long_stats.n_tokens


# ---------------------------------------------------------------------------
# run_sweep — end-to-end with mocked encode
# ---------------------------------------------------------------------------


def test_run_sweep_writes_enriched_jsonl(tmp_path, monkeypatch) -> None:
    _patch_encode(monkeypatch)
    s = _scenario_truncate()
    model, _ = _make_fake_model()
    embedder_tag = "m@rev"
    cfg = em.EncoderConfig(batch_size=8, min_char_length=0, long_doc=None)

    records = [
        _corpus_record(ci_id="a", lg="fr", year=1920, provider="P", alias="aA"),
        _corpus_record(ci_id="b", lg="de", year=1930, provider="P", alias="aA"),
    ]

    out_path = tmp_path / "out.jsonl.bz2"
    stats = embed_sweep.run_sweep(
        scenario=s,
        corpus_records=records,
        model=model,
        encoder_cfg=cfg,
        embedder_tag=embedder_tag,
        local_output=out_path,
    )

    with bz2.open(out_path, "rb") as fh:
        emitted = [orjson.loads(line) for line in fh if line.strip()]

    assert stats.embedded == 2
    assert {r["ci_id"] for r in emitted} == {"a", "b"}
    a = next(r for r in emitted if r["ci_id"] == "a")
    # production schema
    assert a["model_id"] == embedder_tag
    assert isinstance(a["embedding"], list)
    assert a["size"] == len(a["embedding"])
    # research metadata merged in
    assert a["lg"] == "fr"
    assert a["year"] == 1920
    assert a["provider"] == "P"
    assert a["alias"] == "aA"
    assert a["len_chars"] == 20000
    assert a["ocrqa"] == 0.95
    # scenario annotations
    assert a["scenario_id"] == "S0"
    assert a["chunker"] == "truncate"
    assert a["chunk_tokens"] is None
    assert a["n_chunks"] == 1
    # token counts: S0 is one-shot, so n_tokens == n_tokens_per_chunk[0]
    # and equals len(ft) under the fake 1-token-per-char tokenizer.
    expected_tokens = len(records[0]["ft"])
    assert a["n_tokens"] == expected_tokens
    assert a["n_tokens_per_chunk"] == [expected_tokens]
    # study fields are absent when not passed
    assert "study_name" not in a
    assert "study_config_sha" not in a


def test_run_sweep_emits_study_provenance_when_provided(tmp_path, monkeypatch) -> None:
    _patch_encode(monkeypatch)
    s = _scenario_truncate()
    model, _ = _make_fake_model()
    cfg = em.EncoderConfig(batch_size=8, min_char_length=0, long_doc=None)
    out = tmp_path / "out.jsonl.bz2"
    embed_sweep.run_sweep(
        scenario=s,
        corpus_records=[_corpus_record(ci_id="a")],
        model=model,
        encoder_cfg=cfg,
        embedder_tag="m@rev",
        local_output=out,
        study_name="my-study",
        study_config_sha="abc123def456",
    )
    with bz2.open(out, "rb") as fh:
        record = orjson.loads(fh.readline())
    assert record["study_name"] == "my-study"
    assert record["study_config_sha"] == "abc123def456"


def test_run_sweep_propagates_n_chunks_for_chunked_scenario(tmp_path, monkeypatch) -> None:
    _patch_encode(monkeypatch)
    s = _scenario_fixed_window(512)
    model, _ = _make_fake_model()
    long_doc = embed_sweep.build_long_doc_config(s, model)
    cfg = em.EncoderConfig(batch_size=64, min_char_length=0, long_doc=long_doc)

    records = [
        _corpus_record(ci_id="short", ft="x" * 100),
        _corpus_record(ci_id="long", ft="x" * 2000),
    ]
    out_path = tmp_path / "out.jsonl.bz2"
    embed_sweep.run_sweep(
        scenario=s,
        corpus_records=records,
        model=model,
        encoder_cfg=cfg,
        embedder_tag="m@rev",
        local_output=out_path,
    )
    with bz2.open(out_path, "rb") as fh:
        emitted = {orjson.loads(line)["ci_id"]: orjson.loads(line) for line in fh if line.strip()}
    assert emitted["short"]["n_chunks"] == 1
    assert emitted["short"]["n_tokens"] == 100
    assert emitted["short"]["n_tokens_per_chunk"] == [100]
    # 2000 fake tokens / 512-token windows → 4 chunks
    assert emitted["long"]["n_chunks"] == 4
    assert emitted["long"]["chunker"] == "fixed-window"
    assert emitted["long"]["chunk_tokens"] == 512
    assert emitted["long"]["n_tokens"] == 2000
    assert emitted["long"]["n_tokens_per_chunk"] == [512, 512, 512, 464]


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_list_prints_table(capsys) -> None:
    rc = embed_sweep.main(["--config", str(_study_v1_path()), "--list"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "S0" in captured.out
    assert "S15" in captured.out


def test_cli_requires_config() -> None:
    with pytest.raises(SystemExit):
        embed_sweep.main([])


def test_cli_requires_scenario_when_not_listing() -> None:
    with pytest.raises(SystemExit):
        embed_sweep.main(["--config", str(_study_v1_path())])


def test_cli_unknown_scenario_id_errors() -> None:
    with pytest.raises(SystemExit):
        embed_sweep.main(["--config", str(_study_v1_path()), "--scenario", "Sxxx"])


def test_output_key_layout_via_study_config() -> None:
    """Per-scenario shard now lands flat at <s3_root>/<scenario.id>.jsonl.bz2.

    The old <prefix>/<scenario_id>_<label>/<corpus_basename> layout is
    gone — see study_config.scenario_filename().
    """
    from impresso_text_embedder.research.study_config import (
        load_study_config,
        scenario_filename,
    )

    cfg = load_study_config(_study_v1_path())
    s = _scenario_fixed_window(1024, sid="S2")
    assert cfg.s3_key(scenario_filename(s.id)) == "chunking-eval/v1/S2.jsonl.bz2"
    assert cfg.local_path(scenario_filename(s.id)).name == "S2.jsonl.bz2"


def test_read_corpus_round_trip(tmp_path) -> None:
    records = [_corpus_record(ci_id="a"), _corpus_record(ci_id="b")]
    p = tmp_path / "corpus.jsonl.bz2"
    _write_corpus(p, records)
    loaded = embed_sweep._read_corpus(p)
    assert [r["ci_id"] for r in loaded] == ["a", "b"]


# ---------------------------------------------------------------------------
# Logging — sweep-shaped paths under <log-dir>/<date>/<scenario_id>.log
# ---------------------------------------------------------------------------


def test_resolve_research_log_dir_uses_pvc_default() -> None:
    base = embed_sweep._resolve_research_log_dir(None)
    assert base.is_relative_to(embed_sweep.logging_setup.DEFAULT_RCP_SCRATCH)
    assert "chunking-eval" in base.parts
    assert "embeddings" not in base.parts


def test_resolve_research_log_dir_fails_when_pvc_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        embed_sweep.logging_setup,
        "DEFAULT_RCP_SCRATCH",
        tmp_path / "definitely-not-mounted",
    )
    with pytest.raises(SystemExit) as exc:
        embed_sweep._resolve_research_log_dir(None)
    assert "PVC" in str(exc.value) or "mounted" in str(exc.value)


def test_resolve_research_log_dir_passes_through_explicit_override(tmp_path) -> None:
    override = tmp_path / "custom"
    assert embed_sweep._resolve_research_log_dir(override) == override
