"""
This module, `s3_to_local_stamps.py`, is a utility for creating local stamp files that
mirror the structure of specified S3 objects.

The main class, `LocalStampCreator`, orchestrates the process of creating these stamp
files. It uses the boto3 library to interact with the S3 service and the os library to
create local directories and files.

The module uses environment variables to configure the S3 resource object. These
variables include the access key, secret key, and host URL for the S3 service.

The module can be run as a standalone script. It accepts command-line arguments for the
S3 bucket name, file prefix to match in the bucket, verbosity level for logging, and an
optional log file.
"""

__author__ = "simon.clematide@uzh.ch"
__license__ = "GNU GPL 3.0 or later"

import argparse
import datetime
import logging
import os
import sys
from typing import Any, Tuple, Optional
import bz2
import boto3
from smart_open import open
from dotenv import load_dotenv

log = logging.getLogger(__name__)


def get_s3_resource(
    access_key: str | None = None,
    secret_key: str | None = None,
    host_url: str | None = None,
) -> Any:
    """Configures and returns an S3 resource object.

    If the optional access key, secret key, and host URL are not provided, the
    method uses environment variables to configure the S3 resource object. Support
    .env configuration.

    Args:
        access_key (str | None, optional): The access key for S3. Defaults to None.
        secret_key (str | None, optional): The secret key for S3. Defaults to None.
        host_url (str | None, optional): The host URL for S3. Defaults to None.

        Returns:
            Any: The configured S3 resource.
    """

    load_dotenv()
    access_key = access_key or os.getenv("SE_ACCESS_KEY")
    secret_key = secret_key or os.getenv("SE_SECRET_KEY")
    host_url = host_url or os.getenv("SE_HOST_URL")
    return boto3.resource(
        "s3",
        aws_secret_access_key=secret_key,
        aws_access_key_id=access_key,
        endpoint_url=host_url,
    )


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parses an S3 path into a bucket name and prefix.

    Args:
        s3_path (str): The S3 path to parse.

    Returns:
        Tuple[str, str]: The bucket name and prefix.

    Raises:
        ValueError: If the S3 path does not start with "s3://" or if it does not include both a bucket name and prefix.

    >>> parse_s3_path("s3://mybucket/myfolder/myfile.txt")
    ('mybucket', 'myfolder/myfile.txt')

    >>> parse_s3_path("s3://mybucket/myfolder/subfolder/")
    ('mybucket', 'myfolder/subfolder/')

    >>> parse_s3_path("not-an-s3-path")
    Traceback (most recent call last):
    ...
    ValueError: S3 path must start with s3://
    """
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")
    path_parts = s3_path[5:].split("/", 1)
    if len(path_parts) < 2:
        raise ValueError("S3 path must include both bucket name and prefix")
    return path_parts[0], path_parts[1]


class LocalStampCreator(object):
    """Main application for creating local stamp files mirroring S3 objects.

    Attributes:
        args (Any): Command-line arguments object.
        s3_resource (boto3.resources.factory.s3.ServiceResource): The S3 service resource.

    Methods:
        run(): Orchestrates the stamp file creation process.

        create_stamp_files(bucket_name: str, prefix: str): Creates local stamp files
            based on S3 objects.

        create_local_stamp_file(s3_key: str, last_modified: datetime.datetime): Creates
            a single local stamp file.
    """

    def __init__(self, args: argparse.Namespace):
        """Initializes the application with command-line arguments.

        Args:
            args: Command-line arguments.
        """

        self.args = args
        self.s3_resource = get_s3_resource()
        self.stats = {"files_created": 0}  # Initialize the statistics dictionary
        # Splitting the s3-path into bucket name and prefix
        self.bucket_name, self.prefix = parse_s3_path(self.args.s3_path)

    def run(self) -> None:
        """Orchestrates the stamp file creation process based on S3 objects."""

        log.info("Starting stamp file creation...")
        self.create_stamp_files(self.bucket_name, self.prefix)
        log.info(
            "Stamp file creation completed. Files created:"
            f" {self.stats['files_created']}"
        )

    def create_stamp_files(self, bucket_name: str, prefix: str) -> None:
        """Creates local stamp files that mirror the structure of specified S3 objects.

        Args:
            bucket_name (str): The name of the S3 bucket.
            prefix (str): The file prefix to match in the S3 bucket.
        """
        bucket = self.s3_resource.Bucket(bucket_name)
        for s3_object in bucket.objects.filter(Prefix=prefix):

            s3_key = s3_object.key
            last_modified = s3_object.last_modified

            # Get the content of the S3 object
            content = (
                self.get_s3_object_content(s3_key) if self.args.write_content else None
            )

            # Create a local stamp file
            self.create_local_stamp_file(s3_key, last_modified, content)

    def get_s3_object_content(self, s3_key: str) -> str:
        """Get the content of an S3 object.

        Args:
            s3_key (str): The key of the S3 object.

        Returns:
            str: The content of the S3 object.
        """

        obj = self.s3_resource.Object(self.bucket_name, s3_key)
        compressed_content = obj.get()["Body"].read()

        # Decompress the content
        if s3_key.endswith(".bz2"):
            decompressed_content = bz2.decompress(compressed_content)

        return decompressed_content.decode("utf-8")

    def create_local_stamp_file(
        self,
        s3_key: str,
        last_modified: datetime.datetime,
        content: Optional[str] = None,
    ) -> None:
        """Creates a local stamp file, mirroring the modification date of an S3 object.

        Args:
            s3_key (str): The key of the S3 object.

            last_modified (datetime.datetime): The last-modified timestamp of the S3
                 object.
        """

        local_file_path = s3_key.replace("/", os.sep)
        # include  bucket name in local file path depending on the --no-bucket flag
        if not self.args.no_bucket:
            local_file_path = os.path.join(self.bucket_name, local_file_path)

        # Adjust the file path to include the local directory
        local_file_path = os.path.join(self.args.local_dir, local_file_path)
        if content is None:
            local_file_path += self.args.stamp_extension

        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        with open(local_file_path, "w", encoding="utf-8") as f:
            f.write(content if content is not None else "")

        os.utime(
            local_file_path, (last_modified.timestamp(), last_modified.timestamp())
        )

        self.stats["files_created"] += 1
        log.info(
            f"Created stamp file: '{local_file_path}' with modification date:"
            f" {last_modified}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="S3 to Local Stamp File Creator",
        epilog="Utility to mirror S3 file structure locally with stamp files.",
    )

    parser.add_argument(
        "s3_path",
        help=(
            "S3 path prefix in the format s3://BUCKET_NAME/PREFIX. "
            "The prefix is used to match objects in the specified bucket."
        ),
    )
    parser.add_argument(
        "--level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default: %(default)s",
    )
    parser.add_argument("--logfile", help="Write log to FILE", metavar="FILE")

    parser.add_argument(
        "--local-dir",
        default="./",
        type=str,
        help="Local directory prefix for creating stamp files %(default)s",
    )
    parser.add_argument(
        "--no-bucket",
        help="Do not use bucket name for local files, only the key. %(default)s",
        action="store_true",
    )
    parser.add_argument(
        "--stamp-extension",
        help=(
            "Append this extension to all file names created (preceding dot must be"
            " specified). %(default)s"
        ),
        default=".stamp",
    )
    parser.add_argument(
        "--write-content",
        action="store_true",
        help="Write the content of the S3 objects to the stamp files.",
    )
    arguments = parser.parse_args()

    to_logging_level = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    logging.basicConfig(
        level=to_logging_level[arguments.level],
        format="%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s",
        force=True,
    )
    try:
        processor = LocalStampCreator(arguments)
        processor.run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
