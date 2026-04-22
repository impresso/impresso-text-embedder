from __future__ import annotations

import bz2
import io as _io
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from impresso_text_embedder import io


class TestParseS3Uri:
    def test_basic(self):
        assert io.parse_s3_uri("s3://bucket/key/file.txt") == ("bucket", "key/file.txt")

    def test_nested_key(self):
        assert io.parse_s3_uri("s3://b/a/b/c.bz2") == ("b", "a/b/c.bz2")

    @pytest.mark.parametrize(
        "bad",
        ["http://x/y", "s3:/bucket/key", "s3://bucket", "s3:///key", "s3://bucket/"],
    )
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            io.parse_s3_uri(bad)


class TestBuildOutputKey:
    def test_shape(self):
        assert (
            io.build_output_key("SNL", "EXP", 1912, "gte-multilingual-base")
            == "embeddings/docs/gte-multilingual-base/SNL/EXP/EXP-1912.jsonl.bz2"
        )


class TestParseInputKey:
    def test_no_prefix(self):
        parsed = io.parse_input_key("SNL/EXP/EXP-1912.jsonl.bz2")
        assert parsed.provider == "SNL"
        assert parsed.alias == "EXP"
        assert parsed.year == 1912

    def test_with_prefix(self):
        parsed = io.parse_input_key(
            "lingproc/lingproc-test-v1.0.0/SNL/EXP/EXP-1912.jsonl.bz2",
            input_prefix="lingproc/lingproc-test-v1.0.0",
        )
        assert parsed == io.InputKey(
            "SNL", "EXP", 1912, "lingproc/lingproc-test-v1.0.0/SNL/EXP/EXP-1912.jsonl.bz2"
        )

    def test_prefix_trailing_slash_tolerated(self):
        parsed = io.parse_input_key(
            "p/SNL/EXP/EXP-1912.jsonl.bz2", input_prefix="p/"
        )
        assert parsed.provider == "SNL"

    def test_alias_mismatch(self):
        with pytest.raises(ValueError, match="disagrees"):
            io.parse_input_key("SNL/EXP/OTHER-1912.jsonl.bz2")

    @pytest.mark.parametrize(
        "bad",
        [
            "SNL/EXP/EXP-abcd.jsonl.bz2",  # non-numeric year
            "SNL/EXP/EXP-1912.jsonl",  # wrong extension
            "SNL/EXP/EXP.jsonl.bz2",  # missing year
            "too/shallow.jsonl.bz2",  # not enough path parts
        ],
    )
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            io.parse_input_key(bad)

    def test_prefix_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not start with prefix"):
            io.parse_input_key("A/SNL/EXP/EXP-1912.jsonl.bz2", input_prefix="B")


class TestListInputKeys:
    def _make_s3_client(self, keys):
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": k} for k in keys]},
        ]
        client = MagicMock()
        client.get_paginator.return_value = paginator
        return client, paginator

    def test_filters_and_parses(self):
        client, paginator = self._make_s3_client(
            [
                "p/SNL/EXP/EXP-1910.jsonl.bz2",
                "p/SNL/EXP/EXP-1911.jsonl.bz2",
                "p/SNL/EXP/README.txt",  # wrong suffix → skipped
                "p/SNL/GDL/GDL-1911.jsonl.bz2",
            ]
        )
        with patch.object(io, "get_s3_client", return_value=client):
            keys = list(
                io.list_input_keys(
                    "bucket", "SNL", input_prefix="p", alias_filter={"EXP"}, year_min=1911
                )
            )
        paginator.paginate.assert_called_once_with(Bucket="bucket", Prefix="p/SNL/")
        assert [(k.alias, k.year) for k in keys] == [("EXP", 1911)]

    def test_handles_empty_page(self):
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [{}]
        with patch.object(io, "get_s3_client", return_value=client):
            assert list(io.list_input_keys("b", "P")) == []


class TestObjectExists:
    def _client_that_raises(self, code):
        err = ClientError({"Error": {"Code": code}}, "HeadObject")
        client = MagicMock()
        client.head_object.side_effect = err
        return client

    def test_true_when_head_succeeds(self):
        client = MagicMock()
        with patch.object(io, "get_s3_client", return_value=client):
            assert io.object_exists("b", "k") is True

    @pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
    def test_false_on_missing_codes(self, code):
        with patch.object(io, "get_s3_client", return_value=self._client_that_raises(code)):
            assert io.object_exists("b", "k") is False

    def test_reraises_other_errors(self):
        with patch.object(
            io, "get_s3_client", return_value=self._client_that_raises("AccessDenied")
        ):
            with pytest.raises(ClientError):
                io.object_exists("b", "k")


class TestIterJsonlBz2:
    def test_streams_lines(self):
        payload = b"\n".join([b'{"id":1}', b'{"id":2}', b""])  # trailing empty line
        compressed = bz2.compress(payload)
        body = _io.BytesIO(compressed)

        fake_obj = MagicMock()
        fake_obj.get.return_value = {"Body": body}
        fake_resource = MagicMock()
        fake_resource.Object.return_value = fake_obj

        with patch.object(io, "get_s3_resource", return_value=fake_resource):
            lines = list(io.iter_jsonl_bz2("b", "k"))

        assert lines == ['{"id":1}', '{"id":2}']
        fake_resource.Object.assert_called_once_with("b", "k")


class TestUploadLocalFile:
    def test_raises_when_upload_returns_false(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi")
        with patch.object(io, "upload_to_s3", return_value=False):
            with pytest.raises(RuntimeError, match="upload"):
                io.upload_local_file(f, "b", "k")

    def test_silent_on_success(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi")
        with patch.object(io, "upload_to_s3", return_value=True) as mocked:
            io.upload_local_file(f, "b", "k")
        mocked.assert_called_once_with(str(f), "k", "b")
