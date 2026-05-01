from __future__ import annotations

import bz2
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import orjson
import pytest

from impresso_text_embedder.research import query_embed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _query_record(
    *,
    query_id: str = "q1",
    ci_id: str = "doc-1",
    lg: str = "fr",
    query_text: str = "Quel est le sujet de cet article ?",
    query_type: str = "question",
    position_bucket: str = "head",
    position_chars: tuple[int, int] = (0, 1000),
    references: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    rec = {
        "query_id": query_id,
        "ci_id": ci_id,
        "lg": lg,
        "query_text": query_text,
        "query_type": query_type,
        "position_bucket": position_bucket,
        "position_chars": list(position_chars),
        "references": references
        if references is not None
        else [{"text": "verbatim span", "char_start": 12, "char_end": 25}],
        "gen_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "gen_endpoint": "https://inference.rcp.epfl.ch/v1",
        "ts": "2026-01-01T00:00:00Z",
    }
    if extra:
        rec.update(extra)
    return rec


def _write_queries(path: Path, records: list[dict]) -> None:
    with bz2.open(path, "wb") as fh:
        for r in records:
            fh.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))


def _patch_encode(monkeypatch, dim: int = 4) -> None:
    """Replace ``encode_texts`` with a deterministic stub.

    Returns a numpy array shaped ``[len(texts), dim]`` so the
    production ``.tolist()`` path is exercised. Each row encodes the
    text length in slot 0 + a stable signature so different inputs
    produce different vectors.
    """

    def fake(*, model, texts, batch_size, show_progress_bar=False, precision="bf16"):
        return np.asarray(
            [[float(len(t)), float(i), 0.123456789, 0.0] for i, t in enumerate(texts)],
            dtype=np.float32,
        )

    monkeypatch.setattr(query_embed, "encode_texts", fake)


def _study_v1_path() -> Path:
    return (Path(__file__).parent.parent / "configs/research/study-v1.yaml").resolve()


# ---------------------------------------------------------------------------
# _read_queries
# ---------------------------------------------------------------------------


def test_read_queries_round_trip(tmp_path) -> None:
    records = [_query_record(query_id="q1"), _query_record(query_id="q2")]
    p = tmp_path / "queries.jsonl.bz2"
    _write_queries(p, records)
    loaded = query_embed._read_queries(p)
    assert [r["query_id"] for r in loaded] == ["q1", "q2"]


def test_read_queries_skips_blank_lines(tmp_path) -> None:
    p = tmp_path / "queries.jsonl.bz2"
    with bz2.open(p, "wb") as fh:
        fh.write(orjson.dumps(_query_record(query_id="q1"), option=orjson.OPT_APPEND_NEWLINE))
        fh.write(b"\n")
        fh.write(orjson.dumps(_query_record(query_id="q2"), option=orjson.OPT_APPEND_NEWLINE))
    loaded = query_embed._read_queries(p)
    assert [r["query_id"] for r in loaded] == ["q1", "q2"]


# ---------------------------------------------------------------------------
# _round_embedding
# ---------------------------------------------------------------------------


def test_round_embedding_truncates_to_5_decimals() -> None:
    out = query_embed._round_embedding([0.123456789, 1.0, 0.000001])
    assert out == [0.12346, 1.0, 0.0]


# ---------------------------------------------------------------------------
# _build_output_record
# ---------------------------------------------------------------------------


def test_build_output_record_contains_text_record_shape_plus_query_metadata() -> None:
    q = _query_record()
    rec = query_embed._build_output_record(
        q,
        [0.1, 0.2, 0.3, 0.4],
        embedder_tag="m@rev",
        ts="2026-04-30T10:00:00Z",
        study_name="study-v1",
        study_config_sha="abc123def456",
    )
    # Production text-level shape primitives:
    assert rec["ci_id"] == "doc-1"
    assert rec["model_id"] == "m@rev"
    assert rec["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert rec["size"] == 4
    assert rec["ts"] == "2026-04-30T10:00:00Z"
    # Query identity:
    assert rec["query_id"] == "q1"
    # Mirrored query metadata:
    assert rec["lg"] == "fr"
    assert rec["query_type"] == "question"
    assert rec["position_bucket"] == "head"
    assert rec["position_chars"] == [0, 1000]
    assert rec["references"] == [{"text": "verbatim span", "char_start": 12, "char_end": 25}]
    assert rec["query_text"] == "Quel est le sujet de cet article ?"
    # Provenance:
    assert rec["study_name"] == "study-v1"
    assert rec["study_config_sha"] == "abc123def456"
    # Things explicitly NOT mirrored (gen_model/gen_endpoint live on
    # the input shard, not duplicated here):
    assert "gen_model" not in rec
    assert "gen_endpoint" not in rec


def test_build_output_record_omits_provenance_when_unset() -> None:
    rec = query_embed._build_output_record(
        _query_record(),
        [0.0, 0.0],
        embedder_tag="m@rev",
        ts="2026-04-30T10:00:00Z",
        study_name=None,
        study_config_sha=None,
    )
    assert "study_name" not in rec
    assert "study_config_sha" not in rec


# ---------------------------------------------------------------------------
# run_query_embed — end-to-end with mocked encode
# ---------------------------------------------------------------------------


def test_run_query_embed_writes_one_record_per_query(tmp_path, monkeypatch) -> None:
    _patch_encode(monkeypatch, dim=4)
    queries = [
        _query_record(query_id="q1", query_text="alpha"),
        _query_record(query_id="q2", query_text="beta beta"),
    ]
    out = tmp_path / "queries-embedded.jsonl.bz2"
    stats = query_embed.run_query_embed(
        queries=queries,
        model=MagicMock(),
        batch_size=8,
        precision="bf16",
        embedder_tag="m@rev",
        local_output=out,
    )

    with bz2.open(out, "rb") as fh:
        emitted = [orjson.loads(line) for line in fh if line.strip()]

    assert stats.queries_total == 2
    assert stats.embedded == 2
    assert stats.skipped_empty == 0
    assert stats.dim == 4
    assert [r["query_id"] for r in emitted] == ["q1", "q2"]
    assert all(r["model_id"] == "m@rev" for r in emitted)
    # 5-decimal rounding applied:
    assert emitted[0]["embedding"][2] == pytest.approx(0.12346)
    # size == dim:
    assert all(r["size"] == 4 == len(r["embedding"]) for r in emitted)


def test_run_query_embed_skips_empty_query_text(tmp_path, monkeypatch) -> None:
    _patch_encode(monkeypatch)
    queries = [
        _query_record(query_id="q1", query_text="ok"),
        _query_record(query_id="q2", query_text=""),
        _query_record(query_id="q3", query_text="   "),
    ]
    out = tmp_path / "out.jsonl.bz2"
    stats = query_embed.run_query_embed(
        queries=queries,
        model=MagicMock(),
        batch_size=8,
        precision="bf16",
        embedder_tag="m@rev",
        local_output=out,
    )
    assert stats.queries_total == 3
    assert stats.embedded == 1
    assert stats.skipped_empty == 2
    with bz2.open(out, "rb") as fh:
        emitted = [orjson.loads(line) for line in fh if line.strip()]
    assert [r["query_id"] for r in emitted] == ["q1"]


def test_run_query_embed_writes_empty_file_on_empty_input(tmp_path, monkeypatch) -> None:
    _patch_encode(monkeypatch)
    out = tmp_path / "out.jsonl.bz2"
    stats = query_embed.run_query_embed(
        queries=[],
        model=MagicMock(),
        batch_size=8,
        precision="bf16",
        embedder_tag="m@rev",
        local_output=out,
    )
    assert stats.embedded == 0
    assert stats.dim is None
    # File exists and is a valid empty bz2 (so downstream consumers
    # don't have to special-case its absence).
    assert out.exists()
    with bz2.open(out, "rb") as fh:
        assert fh.read() == b""


def test_run_query_embed_emits_study_provenance(tmp_path, monkeypatch) -> None:
    _patch_encode(monkeypatch)
    out = tmp_path / "out.jsonl.bz2"
    query_embed.run_query_embed(
        queries=[_query_record()],
        model=MagicMock(),
        batch_size=8,
        precision="bf16",
        embedder_tag="m@rev",
        local_output=out,
        study_name="study-v1",
        study_config_sha="abc123def456",
    )
    with bz2.open(out, "rb") as fh:
        record = orjson.loads(fh.readline())
    assert record["study_name"] == "study-v1"
    assert record["study_config_sha"] == "abc123def456"


def test_run_query_embed_preserves_mirrored_fields_verbatim(tmp_path, monkeypatch) -> None:
    """Even unusual values (Unicode, nested refs) round-trip untouched."""
    _patch_encode(monkeypatch)
    refs = [
        {"text": "Zürich, l'« Aufklärung »", "char_start": 100, "char_end": 200},
        {"text": "second", "char_start": 500, "char_end": 510},
    ]
    q = _query_record(
        query_id="q-de-1",
        ci_id="doc-de",
        lg="de",
        query_text="Wie hieß der Bürgermeister?",
        query_type="topical-phrase",
        position_bucket="tail",
        position_chars=(40000, 60000),
        references=refs,
    )
    out = tmp_path / "out.jsonl.bz2"
    query_embed.run_query_embed(
        queries=[q],
        model=MagicMock(),
        batch_size=8,
        precision="bf16",
        embedder_tag="m@rev",
        local_output=out,
    )
    with bz2.open(out, "rb") as fh:
        rec = orjson.loads(fh.readline())
    assert rec["lg"] == "de"
    assert rec["query_type"] == "topical-phrase"
    assert rec["position_bucket"] == "tail"
    assert rec["position_chars"] == [40000, 60000]
    assert rec["references"] == refs
    assert rec["query_text"] == "Wie hieß der Bürgermeister?"


def test_run_query_embed_accepts_list_of_lists_from_encode(tmp_path, monkeypatch) -> None:
    """`encode_texts` may return either an ndarray or a plain list-of-lists.

    The runner must handle both — a future cache layer or alternate
    embedder may not return numpy.
    """

    def fake(*, model, texts, batch_size, show_progress_bar=False, precision="bf16"):
        return [[float(i), 0.0, 0.0, 0.0] for i, _ in enumerate(texts)]

    monkeypatch.setattr(query_embed, "encode_texts", fake)
    out = tmp_path / "out.jsonl.bz2"
    stats = query_embed.run_query_embed(
        queries=[_query_record(query_id="q1"), _query_record(query_id="q2")],
        model=MagicMock(),
        batch_size=8,
        precision="bf16",
        embedder_tag="m@rev",
        local_output=out,
    )
    assert stats.embedded == 2
    assert stats.dim == 4


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_requires_config() -> None:
    with pytest.raises(SystemExit):
        query_embed.main([])


def test_cli_rejects_unknown_flag() -> None:
    with pytest.raises(SystemExit):
        query_embed.main(["--config", str(_study_v1_path()), "--bogus"])


# ---------------------------------------------------------------------------
# _format_stats
# ---------------------------------------------------------------------------


def test_format_stats_includes_dim_when_known() -> None:
    s = query_embed.EmbedStats(queries_total=10, embedded=10, dim=768)
    line = query_embed._format_stats(s, embedder_tag="m@rev")
    assert "queries_total=10" in line
    assert "embedded=10" in line
    assert "dim=768" in line
    assert "model=m@rev" in line


def test_format_stats_includes_skipped_only_when_nonzero() -> None:
    s = query_embed.EmbedStats(queries_total=3, embedded=2, skipped_empty=1, dim=4)
    assert "skipped_empty=1" in query_embed._format_stats(s, embedder_tag="m@rev")
    s2 = query_embed.EmbedStats(queries_total=3, embedded=3, skipped_empty=0, dim=4)
    assert "skipped_empty" not in query_embed._format_stats(s2, embedder_tag="m@rev")


# ---------------------------------------------------------------------------
# Logging — same fail-fast behaviour as embed_sweep
# ---------------------------------------------------------------------------


def test_resolve_research_log_dir_uses_pvc_default_when_mounted() -> None:
    base = query_embed._resolve_research_log_dir(None)
    assert base.is_relative_to(query_embed.logging_setup.DEFAULT_RCP_SCRATCH)
    assert "chunking-eval" in base.parts
    assert "embeddings" not in base.parts


def test_resolve_research_log_dir_fails_when_pvc_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        query_embed.logging_setup,
        "DEFAULT_RCP_SCRATCH",
        tmp_path / "definitely-not-mounted",
    )
    with pytest.raises(SystemExit) as exc:
        query_embed._resolve_research_log_dir(None)
    assert "PVC" in str(exc.value) or "mounted" in str(exc.value)


def test_resolve_research_log_dir_passes_through_explicit_override(tmp_path) -> None:
    override = tmp_path / "custom"
    assert query_embed._resolve_research_log_dir(override) == override


# ---------------------------------------------------------------------------
# Output S3 layout via study config
# ---------------------------------------------------------------------------


def test_output_key_layout_via_study_config() -> None:
    """``queries-embedded.jsonl.bz2`` lands flat at the study's S3 prefix.

    Asserts the path convention so a renamed constant or a malformed
    path template is caught at test time, not at submit time.
    """
    from impresso_text_embedder.research.study_config import (
        QUERIES_EMBEDDED_FILENAME,
        load_study_config,
    )

    cfg = load_study_config(_study_v1_path())
    assert cfg.s3_key(QUERIES_EMBEDDED_FILENAME) == (
        "chunking-eval/v1/queries-embedded.jsonl.bz2"
    )
    assert cfg.local_path(QUERIES_EMBEDDED_FILENAME).name == "queries-embedded.jsonl.bz2"
