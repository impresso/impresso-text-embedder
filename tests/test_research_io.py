"""Unit tests for ``research._io`` — the staged_input / staged_output
context managers that replaced the per-CLI tempfile-and-cleanup
plumbing in the four research CLIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from impresso_text_embedder.research import _io as research_io


class TestStagedInput:
    def test_downloads_to_tempfile_and_cleans_up(self, monkeypatch, tmp_path: Path):
        """staged_input downloads to a tempfile and removes it on exit."""
        captured: dict = {}

        def fake_download(bucket: str, key: str, dest):
            dest_path = Path(dest)
            captured["bucket"] = bucket
            captured["key"] = key
            captured["dest"] = dest_path
            dest_path.write_bytes(b"payload")

        monkeypatch.setattr(research_io.s3io, "download_to_local", fake_download)

        with research_io.staged_input("buck", "k/v.jsonl.bz2") as path:
            assert path.exists()
            assert path.read_bytes() == b"payload"
            assert path == captured["dest"]

        assert not captured["dest"].exists(), "tempfile should be removed on exit"
        assert captured["bucket"] == "buck"
        assert captured["key"] == "k/v.jsonl.bz2"

    def test_cleans_up_on_exception(self, monkeypatch):
        """A failure inside the with block must still unlink the tempfile."""
        captured: dict = {}

        def fake_download(bucket, key, dest):
            dest_path = Path(dest)
            captured["dest"] = dest_path
            dest_path.write_bytes(b"")

        monkeypatch.setattr(research_io.s3io, "download_to_local", fake_download)

        with pytest.raises(RuntimeError, match="boom"):
            with research_io.staged_input("buck", "k.jsonl.bz2"):
                raise RuntimeError("boom")

        assert not captured["dest"].exists()


class TestStagedOutput:
    def test_upload_path_writes_tempfile_and_uploads(self, monkeypatch):
        captured: dict = {}

        def fake_upload(local_path, bucket, key):
            captured["local_path"] = Path(local_path)
            captured["bucket"] = bucket
            captured["key"] = key
            assert captured["local_path"].exists()
            assert captured["local_path"].read_bytes() == b"data"

        monkeypatch.setattr(research_io.s3io, "upload_local_file", fake_upload)

        # The mirror path must NOT be touched on the upload path.
        mirror = Path("/tmp/never-touched-by-this-test/should-not-exist.jsonl.bz2")
        with research_io.staged_output(
            "bucket", "k/x.jsonl.bz2", mirror, upload=True
        ) as out:
            out.write_bytes(b"data")

        assert captured["bucket"] == "bucket"
        assert captured["key"] == "k/x.jsonl.bz2"
        assert not captured["local_path"].exists(), "tempfile should be cleaned up"
        assert not mirror.exists(), "mirror should be untouched on upload path"

    def test_no_upload_writes_to_mirror_and_keeps_it(self, monkeypatch, tmp_path):
        """upload=False writes to the mirror path and skips uploading."""
        upload_called: list = []
        monkeypatch.setattr(
            research_io.s3io,
            "upload_local_file",
            lambda *a, **kw: upload_called.append((a, kw)),
        )

        mirror = tmp_path / "study-x" / "queries.jsonl.bz2"
        assert not mirror.parent.exists()

        with research_io.staged_output(
            "bucket", "k/x.jsonl.bz2", mirror, upload=False
        ) as out:
            assert out == mirror
            out.write_bytes(b"hello")

        assert mirror.read_bytes() == b"hello"
        assert mirror.parent.is_dir(), "parent dir should be created on entry"
        assert upload_called == []

    def test_upload_aborted_on_body_exception(self, monkeypatch, tmp_path):
        """Exception inside the with block must skip the upload."""
        upload_called: list = []
        monkeypatch.setattr(
            research_io.s3io,
            "upload_local_file",
            lambda *a, **kw: upload_called.append((a, kw)),
        )

        mirror = tmp_path / "x.jsonl.bz2"
        with pytest.raises(ValueError):
            with research_io.staged_output(
                "bucket", "k.jsonl.bz2", mirror, upload=True
            ) as out:
                out.write_bytes(b"partial")
                raise ValueError("boom")

        assert upload_called == [], "upload must NOT happen when body raised"
