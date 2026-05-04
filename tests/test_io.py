from __future__ import annotations

import bz2
import hashlib
import io as _io
from datetime import datetime, timezone
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
    def _make_s3_client(self, contents):
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": contents}]
        client = MagicMock()
        client.get_paginator.return_value = paginator
        return client, paginator

    def test_filters_and_parses(self):
        client, paginator = self._make_s3_client(
            [
                {"Key": "p/SNL/EXP/EXP-1910.jsonl.bz2"},
                {"Key": "p/SNL/EXP/EXP-1911.jsonl.bz2"},
                {"Key": "p/SNL/EXP/README.txt"},  # wrong suffix → skipped
                {"Key": "p/SNL/GDL/GDL-1911.jsonl.bz2"},
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

    def test_populates_last_modified(self):
        when = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        client, _ = self._make_s3_client(
            [{"Key": "SNL/EXP/EXP-1912.jsonl.bz2", "LastModified": when}]
        )
        with patch.object(io, "get_s3_client", return_value=client):
            (only,) = list(io.list_input_keys("bucket", "SNL"))
        assert only.last_modified == when

    def test_handles_empty_page(self):
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [{}]
        with patch.object(io, "get_s3_client", return_value=client):
            assert list(io.list_input_keys("b", "P")) == []


class TestHeadLastModified:
    def _client_with_response(self, resp):
        client = MagicMock()
        client.head_object.return_value = resp
        return client

    def _client_that_raises(self, code):
        err = ClientError({"Error": {"Code": code}}, "HeadObject")
        client = MagicMock()
        client.head_object.side_effect = err
        return client

    def test_returns_last_modified_on_success(self):
        when = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        client = self._client_with_response({"LastModified": when})
        with patch.object(io, "get_s3_client", return_value=client):
            assert io.head_last_modified("b", "k") == when

    @pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
    def test_returns_none_on_missing_codes(self, code):
        with patch.object(io, "get_s3_client", return_value=self._client_that_raises(code)):
            assert io.head_last_modified("b", "k") is None

    def test_reraises_other_errors(self):
        with patch.object(
            io, "get_s3_client", return_value=self._client_that_raises("AccessDenied")
        ):
            with pytest.raises(ClientError):
                io.head_last_modified("b", "k")


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


class TestIterJsonlBz2Path:
    def test_streams_lines_from_local_file(self, tmp_path):
        path = tmp_path / "f.jsonl.bz2"
        with bz2.open(path, "wt", encoding="utf-8") as fh:
            fh.write('{"id":1}\n{"id":2}\n\n')  # trailing blank line
        assert list(io.iter_jsonl_bz2_path(path)) == ['{"id":1}', '{"id":2}']

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "f.jsonl.bz2"
        with bz2.open(path, "wt", encoding="utf-8") as fh:
            fh.write('{"x":1}\n')
        assert list(io.iter_jsonl_bz2_path(str(path))) == ['{"x":1}']


class TestDownloadToLocal:
    def _fake_resource(self):
        bucket = MagicMock()
        resource = MagicMock()
        resource.Bucket.return_value = bucket
        return resource, bucket

    def test_calls_download_file_with_default_config(self, tmp_path):
        resource, bucket = self._fake_resource()
        dest = tmp_path / "out.jsonl.bz2"
        with patch.object(io, "get_s3_resource", return_value=resource):
            io.download_to_local("b", "k/v.jsonl.bz2", dest)
        resource.Bucket.assert_called_once_with("b")
        args, kwargs = bucket.download_file.call_args
        assert args == ("k/v.jsonl.bz2", str(dest))
        assert kwargs["Config"] is io.DEFAULT_TRANSFER_CONFIG

    def test_accepts_custom_transfer_config(self, tmp_path):
        from boto3.s3.transfer import TransferConfig

        resource, bucket = self._fake_resource()
        custom = TransferConfig(max_concurrency=2)
        with patch.object(io, "get_s3_resource", return_value=resource):
            io.download_to_local("b", "k", tmp_path / "x", transfer_config=custom)
        _, kwargs = bucket.download_file.call_args
        assert kwargs["Config"] is custom

    def test_propagates_errors(self, tmp_path):
        resource, bucket = self._fake_resource()
        bucket.download_file.side_effect = OSError("no route to host")
        with patch.object(io, "get_s3_resource", return_value=resource):
            with pytest.raises(OSError, match="no route"):
                io.download_to_local("b", "k", tmp_path / "x")


class TestUploadLocalFile:
    @staticmethod
    def _make_file(tmp_path, name="x.txt", payload=b"hi"):
        f = tmp_path / name
        f.write_bytes(payload)
        md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        return f, md5, len(payload)

    @staticmethod
    def _fakes(upload_side_effect=None, head_response=None, head_side_effect=None):
        """Build a (resource, bucket, client) triple of MagicMocks."""
        fake_bucket = MagicMock()
        if upload_side_effect is not None:
            fake_bucket.upload_file.side_effect = upload_side_effect
        fake_resource = MagicMock()
        fake_resource.Bucket.return_value = fake_bucket
        fake_client = MagicMock()
        if head_side_effect is not None:
            fake_client.head_object.side_effect = head_side_effect
        elif head_response is not None:
            fake_client.head_object.return_value = head_response
        return fake_resource, fake_bucket, fake_client

    @staticmethod
    def _patch(fake_resource, fake_client):
        return patch.multiple(
            io,
            get_s3_resource=MagicMock(return_value=fake_resource),
            get_s3_client=MagicMock(return_value=fake_client),
        )

    def test_raises_when_upload_fails(self, tmp_path):
        f, _, _ = self._make_file(tmp_path)
        fake_resource, _, fake_client = self._fakes(
            upload_side_effect=OSError("boom")
        )
        with self._patch(fake_resource, fake_client):
            with pytest.raises(RuntimeError, match="upload"):
                io.upload_local_file(f, "b", "k")
        # upload failed before HEAD could run
        fake_client.head_object.assert_not_called()

    def test_success_single_part_verifies_etag(self, tmp_path):
        f, md5, size = self._make_file(tmp_path)
        fake_resource, fake_bucket, fake_client = self._fakes(
            head_response={"ContentLength": size, "ETag": f'"{md5}"'}
        )
        with self._patch(fake_resource, fake_client):
            io.upload_local_file(f, "b", "k")
        fake_resource.Bucket.assert_called_once_with("b")
        fake_bucket.upload_file.assert_called_once_with(str(f), "k")
        fake_client.head_object.assert_called_once_with(Bucket="b", Key="k")
        fake_client.delete_object.assert_not_called()

    def test_success_multipart_skips_etag_check(self, tmp_path):
        f, md5, size = self._make_file(tmp_path)
        # Multipart ETag has a `-N` suffix; the hex before the dash is NOT the
        # local MD5, but we should accept it and only verify size.
        fake_resource, _, fake_client = self._fakes(
            head_response={"ContentLength": size, "ETag": '"deadbeef-4"'}
        )
        with self._patch(fake_resource, fake_client):
            io.upload_local_file(f, "b", "k")
        fake_client.delete_object.assert_not_called()
        # Sanity: the local MD5 is not used for comparison in the multipart path
        assert md5 != "deadbeef"

    def test_size_mismatch_raises_and_deletes(self, tmp_path):
        f, md5, size = self._make_file(tmp_path)
        fake_resource, _, fake_client = self._fakes(
            head_response={"ContentLength": size + 1, "ETag": f'"{md5}"'}
        )
        with self._patch(fake_resource, fake_client):
            with pytest.raises(RuntimeError, match="size mismatch"):
                io.upload_local_file(f, "b", "k")
        fake_client.delete_object.assert_called_once_with(Bucket="b", Key="k")

    def test_etag_mismatch_single_part_raises_and_deletes(self, tmp_path):
        f, _, size = self._make_file(tmp_path)
        fake_resource, _, fake_client = self._fakes(
            head_response={"ContentLength": size, "ETag": '"00000000000000000000000000000000"'}
        )
        with self._patch(fake_resource, fake_client):
            with pytest.raises(RuntimeError, match="ETag mismatch"):
                io.upload_local_file(f, "b", "k")
        fake_client.delete_object.assert_called_once_with(Bucket="b", Key="k")

    def test_head_failure_raises_and_attempts_delete(self, tmp_path):
        f, _, _ = self._make_file(tmp_path)
        err = ClientError({"Error": {"Code": "500", "Message": "boom"}}, "HeadObject")
        fake_resource, _, fake_client = self._fakes(head_side_effect=err)
        with self._patch(fake_resource, fake_client):
            with pytest.raises(RuntimeError, match="HEAD"):
                io.upload_local_file(f, "b", "k")
        fake_client.delete_object.assert_called_once_with(Bucket="b", Key="k")

    def test_delete_failure_does_not_shadow_primary_error(self, tmp_path):
        f, md5, size = self._make_file(tmp_path)
        fake_resource, _, fake_client = self._fakes(
            head_response={"ContentLength": size + 1, "ETag": f'"{md5}"'}
        )
        fake_client.delete_object.side_effect = OSError("cleanup died")
        with self._patch(fake_resource, fake_client):
            with pytest.raises(RuntimeError, match="size mismatch"):
                io.upload_local_file(f, "b", "k")

    def test_strips_s3_prefix_from_key(self, tmp_path):
        f, md5, size = self._make_file(tmp_path)
        fake_resource, fake_bucket, fake_client = self._fakes(
            head_response={"ContentLength": size, "ETag": f'"{md5}"'}
        )
        with self._patch(fake_resource, fake_client):
            io.upload_local_file(f, "b", "s3://k/leading")
        fake_bucket.upload_file.assert_called_once_with(str(f), "k/leading")
        fake_client.head_object.assert_called_once_with(Bucket="b", Key="k/leading")


class TestCopyS3Object:
    """Server-side copy via boto3 ``copy_object``.

    All tests mock the boto3 client; no real S3 calls.
    """

    @staticmethod
    def _client(
        *,
        src_bucket: str = "src-b",
        src_key: str = "src/k",
        dst_bucket: str = "dst-b",
        dst_key: str = "dst/k",
        dst_exists: bool = False,
        src_size: int = 1024,
        dst_size_after_copy: int | None = None,
        copy_side_effect: BaseException | None = None,
        src_head_side_effect: BaseException | None = None,
    ) -> MagicMock:
        """Build a client that routes head_object on (Bucket, Key) instead of call ordinal.

        Tracks how many times the dst is HEAD-probed so post-copy
        size-parity vs pre-existence checks can be told apart.
        """
        client = MagicMock()
        copy_done = {"yes": False}

        def head(Bucket, Key):
            if Bucket == src_bucket and Key == src_key:
                if src_head_side_effect is not None:
                    raise src_head_side_effect
                return {"ContentLength": src_size}
            if Bucket == dst_bucket and Key == dst_key:
                # After copy_object has fired, every dst HEAD is a
                # post-copy size-parity probe.
                if copy_done["yes"]:
                    return {
                        "ContentLength": (
                            dst_size_after_copy if dst_size_after_copy is not None
                            else src_size
                        )
                    }
                # Pre-copy: the dst HEAD comes from head_last_modified.
                if dst_exists:
                    return {"LastModified": datetime(2024, 1, 1, tzinfo=timezone.utc)}
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            raise AssertionError(f"unexpected HEAD on s3://{Bucket}/{Key}")

        def copy(**kwargs):
            if copy_side_effect is not None:
                raise copy_side_effect
            copy_done["yes"] = True
            return {}

        client.head_object.side_effect = head
        client.copy_object.side_effect = copy
        return client

    def test_copies_when_dst_missing(self):
        client = self._client(src_size=2048)
        with patch.object(io, "get_s3_client", return_value=client):
            copied = io.copy_s3_object("src-b", "src/k", "dst-b", "dst/k")
        assert copied is True
        client.copy_object.assert_called_once_with(
            CopySource={"Bucket": "src-b", "Key": "src/k"},
            Bucket="dst-b",
            Key="dst/k",
        )

    def test_skips_when_dst_exists_and_overwrite_false(self):
        client = self._client(dst_exists=True)
        with patch.object(io, "get_s3_client", return_value=client):
            copied = io.copy_s3_object("src-b", "src/k", "dst-b", "dst/k")
        assert copied is False
        client.copy_object.assert_not_called()

    def test_overwrites_when_dst_exists_and_overwrite_true(self):
        client = self._client(dst_exists=True, src_size=4096)
        with patch.object(io, "get_s3_client", return_value=client):
            copied = io.copy_s3_object(
                "src-b", "src/k", "dst-b", "dst/k", overwrite=True
            )
        assert copied is True
        client.copy_object.assert_called_once()

    def test_size_mismatch_after_copy_raises(self):
        client = self._client(src_size=2048, dst_size_after_copy=1024)
        with patch.object(io, "get_s3_client", return_value=client):
            with pytest.raises(RuntimeError, match="size mismatch"):
                io.copy_s3_object("src-b", "src/k", "dst-b", "dst/k")

    def test_missing_source_raises(self):
        err = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        client = self._client(src_head_side_effect=err)
        with patch.object(io, "get_s3_client", return_value=client):
            with pytest.raises(RuntimeError, match="not found"):
                io.copy_s3_object("src-b", "src/k", "dst-b", "dst/k")

    def test_copy_object_failure_propagates_as_runtime(self):
        err = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "CopyObject"
        )
        client = self._client(copy_side_effect=err)
        with patch.object(io, "get_s3_client", return_value=client):
            with pytest.raises(RuntimeError, match="copy_object"):
                io.copy_s3_object("src-b", "src/k", "dst-b", "dst/k")
