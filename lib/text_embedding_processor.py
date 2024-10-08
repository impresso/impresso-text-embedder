#!/usr/bin/env python

"""
This module, `text_embedding_processor.py`, is a utility for processing bzip2 compressed
JSONL files from S3 or local storage. It computes semantic embeddings for each document
and outputs the result as a JSONL file.
"""

__author__ = "simon.clematide@uzh.ch"
__license__ = "GNU GPL 3.0 or later"

import argparse
import bz2
import json
import random
import logging
import os
from math import ceil
import sys
import datetime
from collections import Counter
import time
from smart_open import open
from typing import Any, Dict, Iterator, Tuple, Generator, List
from sentence_transformers import SentenceTransformer

import boto3
from dotenv import load_dotenv

random.seed(42)

log = logging.getLogger(__name__)

JSONType = Dict[str, Any]


def print_log_message_summary(highest_level: str):
    """Prints a summary of the log messages relevant to the specified log level."""
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level_index = log_levels.index(highest_level.upper())

    summary = {
        "INFO": [
            "Arguments: {arguments}: Logs the command-line arguments passed.",
            (
                "Processing file %s: Indicates the start of file processing for a given"
                " file path."
            ),
            "Type of embeddings: %s: Logs the type of the embeddings object.",
            "Processed %s valid texts: Logs after every 100 valid texts are processed.",
            (
                "Model loaded: Indicates that the embedding model has been successfully"
                " loaded."
            ),
            (
                "Reading from S3: %s: Indicates that a file is being read from an S3"
                " bucket."
            ),
            (
                "Finished reading %s lines from S3: %s: Logs after completing reading"
                " from an S3 file."
            ),
            (
                "Uploading %s to s3://%s/%s: Logs when a file is about to be uploaded"
                " to S3."
            ),
            (
                "Successfully uploaded %s to s3://%s/%s: Logs a successful file upload"
                " to S3."
            ),
        ],
        "WARNING": [
            (
                "The file s3://%s/%s already exists. Silently quitting: Warns that a"
                " file already exists in S3 and the process is exiting."
            ),
            (
                "Output path %s exists and --no-overwrite is set: Warns that the output"
                " file already exists and --no-overwrite prevents further processing."
            ),
        ],
        "ERROR": [
            (
                "The file %s was not found: Logs an error when the file to upload"
                " cannot be found locally."
            ),
            "Credentials not available: Indicates that AWS credentials are missing.",
            (
                "S3 output path must start with 's3://': Logs an error when an invalid"
                " S3 path is provided."
            ),
        ],
        "DEBUG": [
            (
                "Computing embedding for ID: {data.get('id')}: Logs that an embedding"
                " is being computed for a specific document ID."
            ),
            (
                "Writing embedding: %s: Logs when an embedding is being written to the"
                " output file."
            ),
            (
                "Computed embedding for ID: {result.get('id')}: Logs after the"
                " embedding for a specific document ID has been computed."
            ),
        ],
    }
    statistics_help = [
        (
            "valid_texts: The number of valid texts processed that meet the minimum"
            " character length."
        ),
        "short_texts: The number of texts that were too short to be processed.",
        "total_time: The total time taken for processing all valid texts.",
        (
            "char_count_bucket_5k:{N}: The number of texts with character counts in"
            " buckets of 5000 characters."
        ),
        (
            "valid_texts_lg:{lang}: The number of valid texts processed for a specific"
            " language."
        ),
        "files_created: The number of output files created during the process.",
        (
            "Average time per valid text: The average time taken to process each valid"
            " text."
        ),
        (
            "skipped_type_{type}: The number of texts skipped due to their type not"
            " matching the allowed content types. For example, `skipped_type_ad` means"
            " texts of type 'ad' were skipped."
        ),
    ]
    log.info(f"Log Message HELP for level {highest_level.upper()}:")
    for level in log_levels[level_index:]:
        if level in summary:
            log.info(f"Log Level: {level}")
            for message in summary[level]:
                log.info(f" - {message}")
    log.info("End of Log Message HELP.")
    log.info("Begin of Statistics HELP:")
    for help_message in statistics_help:
        log.info(f" - {help_message}")
    log.info("End of Statistics HELP.")


class TextEmbeddingProcessor:
    """Processes a bzip2 compressed JSONL file from S3, line by line, and computes
    embeddings."""

    def __init__(self, args: Any):
        """Initializes the file processor with command-line arguments and sets up the
        embedding model."""
        load_dotenv()
        self.args = args
        self.s3_resource = None

        if self.args.input_path.startswith("s3://") or self.args.s3_output_path:
            self.s3_resource = self.get_s3_resource()

        if self.args.s3_output_path and self.args.quit_if_s3_output_exists:
            bucket, key = self.parse_s3_path(self.args.s3_output_path)
            if self.file_exists_in_s3(bucket, key):
                log.warning(
                    f"The file s3://{bucket}/{key} already exists. Silently quitting,"
                    " as requested by the option --quit-if-s3-output-exists."
                )
                sys.exit(0)
        self.model = None
        self.stats = Counter(valid_texts=0, short_texts=0, total_time=0)
        self.last_timestamp = None  # UTC timestamp of the last processed document

    def run(self) -> None:
        """Orchestrates the file processing based on S3 objects or local files."""

        log.info("Processing file %s", self.args.input_path)
        lines = self.read_lines(self.args.input_path)
        embeddings = (self.compute_embeddings(json.loads(line)) for line in lines)
        log.info("Type of embeddings: %s", type(embeddings))
        self.write_embeddings(embeddings)
        log.info("Processing %s completed.", self.args.input_path)

        if self.args.s3_output_path and not self.args.s3_output_dry_run:
            self.upload_file_to_s3(self.args.output_path, self.args.s3_output_path)

            if self.args.keep_timestamp_only:
                self.keep_timestamp_only(self.args.output_path)

        self.log_statistics()

    def log_statistics(self):
        """Logs the statistics and calculates the average time per valid text."""
        for k in sorted(self.stats):
            log.info("Statistics: %s: %s", k, self.stats[k])

        if self.stats["valid_texts"] > 0:
            average_time_per_text = self.stats["total_time"] / self.stats["valid_texts"]
            log.info(
                f"Average time per valid text: {average_time_per_text:.4f} seconds"
            )

    def load_model(self):
        log.info(
            "Loading SentenceTransformer model...%s@%s",
            self.args.model_name,
            self.args.model_revision,
        )
        m = SentenceTransformer(
            self.args.model_name,
            trust_remote_code=True,
            revision=self.args.model_revision,
        )
        log.info("Model loaded.")
        return m

    def read_lines(self, input_path: str) -> Generator[str, None, None] | List[str]:
        """Reads lines from a file, either from S3 or locally, based on the file
        path."""
        if input_path.startswith("s3://"):
            bucket_name, prefix = self.parse_s3_path(input_path)
            bucket = self.s3_resource.Bucket(bucket_name)
            obj = bucket.Object(prefix)
            log.info("Reading from S3: %s", input_path)
            with bz2.open(obj.get()["Body"], "rt") as infile:
                lines = infile.readlines()
                log.info(
                    "Finished reading %s lines from S3: %s", len(lines), input_path
                )
                return lines
        else:
            with open(input_path, "rt") as infile:
                return (line for line in infile)

    def compute_embeddings(self, data: JSONType) -> JSONType | None:
        """Computes embeddings for the text in the JSON data."""

        content_item_type = data.get("tp")
        embedder = self.args.model_name + "@" + self.args.model_revision

        if content_item_type not in self.args.content_type:
            self.stats[f"skipped_type_{content_item_type}"] += 1
            return None

        if self.model is None:
            # some newspapers do not contain any valid text, therefore avoiding to load
            # the model if not needed
            self.model = self.load_model()
        log.debug(f"Computing embedding for ID: {data.get('id')}")
        text = data.get("ft", "")
        textlen = len(text)
        if text and textlen > self.args.min_char_length:

            self.stats[f"char_count_bucket_5k:{ceil(textlen / 5000) * 5000}"] += 1

            self.stats["valid_texts"] += 1
            self.stats[f"valid_texts_lg:{data.get('lang')}"] += 1
            start_time = time.time()  # Start timing
            embedding = self.model.encode(
                text,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.args.normalize_embeddings,
            )
            end_time = time.time()  # End timing
            self.stats["total_time"] += (
                end_time - start_time
            )  # Accumulate processing time

            self.last_timestamp = datetime.datetime.fromtimestamp(
                end_time, tz=datetime.timezone.utc
            ).replace(microsecond=0)

            if self.stats["valid_texts"] % 100 == 0:
                log.info(f"Processed {self.stats['valid_texts']} valid texts.")

            result = {
                "id": data.get("id"),
                "ts": self.last_timestamp.isoformat() + "Z",
                "embedder": embedder,
                "len": textlen,
            }

            if self.args.include_text:
                result["text"] = text

            result["embedding"] = [round(n, 5) for n in embedding.tolist()]

            log.debug(f"Computed embedding for ID: {result.get('id')}")
            return result
        else:
            self.stats["short_texts"] += 1

        return None

    def write_embeddings(self, embeddings: Iterator[JSONType]) -> None:
        """Writes computed embeddings to the output file in JSON format."""
        output_file_path = self.args.output_path
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for embedding in embeddings:

                if not embedding:
                    continue
                log.debug("Writing embedding: %s", embedding.get("id"))
                outfile.write(
                    json.dumps(
                        embedding,
                        indent=None,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self.stats["files_created"] += 1

    def file_exists_in_s3(self, bucket: str, key: str) -> bool:
        """Check if a file exists in an S3 bucket."""
        try:
            self.s3_resource.Object(bucket, key).load()
            return True
        except self.s3_resource.meta.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise

    def upload_file_to_s3(self, local_file_path: str, s3_path: str) -> None:
        """Uploads a local file to an S3 bucket if it doesn't already exist."""
        bucket, key = self.parse_s3_path(s3_path)
        if self.file_exists_in_s3(bucket, key):
            log.warning(
                f"The file s3://{bucket}/{key} already exists. Skipping upload."
            )
            return

        try:
            log.info(f"Uploading {local_file_path} to s3://{bucket}/{key}")
            self.s3_resource.Bucket(bucket).upload_file(local_file_path, key)
            log.info(f"Successfully uploaded {local_file_path} to s3://{bucket}/{key}")
        except FileNotFoundError:
            log.error(f"The file {local_file_path} was not found.")
        except self.s3_resource.meta.client.exceptions.NoCredentialsError:
            log.error("Credentials not available.")
        except self.s3_resource.meta.client.exceptions.PartialCredentialsError:
            log.error("Incomplete credentials provided.")
        except Exception as e:
            log.error(f"An error occurred: {e}")

    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse the S3 path into bucket and key."""
        if not s3_path.startswith("s3://"):
            raise ValueError("S3 path must start with 's3://'")
        path_parts = s3_path[5:].split("/", 1)
        if len(path_parts) != 2:
            raise ValueError("S3 path must be in the format 's3://bucket/key'")
        return path_parts[0], path_parts[1]

    def keep_timestamp_only(self, input_path: str, timestamp: datetime = None) -> None:
        """Truncates the local file to zero length and updates the metadata to the given
        UTC timestamp."""

        try:
            # Truncate the file to zero length
            with open(input_path, "w", encoding="utf-8"):
                # opening with 'w' truncates the file
                log.info(f"Truncating {input_path} and setting its timestamp metadata.")

            # Use the provided timestamp or default to the current UTC time
            if timestamp is None:
                timestamp = self.last_timestamp or datetime.datetime.now(datetime.UTC)

            # Convert the timestamp to a Unix timestamp (seconds since epoch)
            timestamp_epoch = timestamp.timestamp()

            # Update the file's modification and access time to the specified timestamp
            os.utime(input_path, (timestamp_epoch, timestamp_epoch))

            log.info(
                f"File {input_path} has been truncated and its timestamp updated to"
                f" {timestamp.isoformat()}."
            )
        except Exception as e:
            log.error(f"Failed to truncate {input_path}: {e}")

    def get_s3_resource(self) -> boto3.resource:
        """Configures and returns an S3 resource object based on environment
        variables."""

        access_key = os.getenv("SE_ACCESS_KEY")
        secret_key = os.getenv("SE_SECRET_KEY")
        host_url = os.getenv("SE_HOST_URL", "https://os.zhdk.cloud.switch.ch/")
        return boto3.resource(
            "s3",
            aws_secret_access_key=secret_key,
            aws_access_key_id=access_key,
            endpoint_url=host_url,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Process a bzip2 compressed JSONL file from S3 or local, compute"
            " embeddings, and output JSON."
        )
    )
    parser.add_argument(
        "--input-path",
        help="S3 path in the format s3://BUCKET/PATH or local path to bzip2 JSONL file",
        required=True,
    )
    parser.add_argument("--output-path", help="Output file path", required=True)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help=(
            "Do not overwrite output file if it exists, no processing is done. No error"
            " is raised, only a warning is logged! This prevents accidental overwrite"
            " or recomputation if a local stamp exists. Defaults: %(default)s"
        ),
    )
    parser.add_argument(
        "--s3-output-path",
        help=(
            "Upload local output file to corresponding s3 bucket after processing. If"
            " this value is set to  locally. Defaults: %(default)s"
        ),
    )
    parser.add_argument(
        "--s3-output-dry-run",
        action="store_true",
        help=(
            "Do not upload local output file to corresponding s3 bucket. Even if"
            " --s3-output-path is set.sDefaults: %(default)s"
        ),
    )
    parser.add_argument(
        "--quit-if-s3-output-exists",
        action="store_true",
        help="Quit if the output file already exists in S3. Defaults: %(default)s",
    )
    parser.add_argument(
        "--keep-timestamp-only",
        action="store_true",
        help=(
            "After uploading to S3, keep only the timestamp of the local output file"
            " for data efficiency. Defaults: %(default)s"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Alibaba-NLP/gte-multilingual-base",
        help="Name of the SentenceTransformer model to use for embedding computation.",
    )
    parser.add_argument(
        "--model-revision",
        help=(
            "Revision of the SentenceTransformer model to use for embedding"
            " computation. Defaults: %(default)s"
        ),
    )
    parser.add_argument(
        "--content-type",
        help="Content type of the input file",
        choices=["ar", "page"],
        default=["ar"],
        nargs="+",
    )
    parser.add_argument(
        "--min-char-length",
        type=int,
        default=400,
        help="Minimum character length of the text to be embedded",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        default=False,
        help="Normalize embeddings to unit vectors. Defaults: %(default)s",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help=(
            "Include the text in the output file for debugging purposes Defaults:"
            " %(default)s"
        ),
    )

    parser.add_argument(
        "--level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default: %(default)s",
    )
    parser.add_argument("--logfile", help="Write log to FILE", metavar="FILE")

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

    log.info(f"Arguments: {arguments}")
    if arguments.s3_output_path and not arguments.s3_output_path.startswith("s3://"):
        log.error("S3 output path must start with 's3://'.")
        sys.exit(1)
    if arguments.keep_timestamp_only and not arguments.s3_output_path:
        log.warning(
            "Will not replace output files with time stamp without S3 output path"
            " option --s3-output-path set. Option --keep-timestamp-only is ignored."
        )
    if (
        arguments.quit_if_s3_output_exists and not arguments.s3_output_path
    ):  # pragma: no cover
        log.warning(
            "Option --quit-if-s3-output-exists is ignored without S3 output path"
            " option --s3-output-path set."
        )
    if (
        arguments.output_path
        and arguments.no_overwrite
        and os.path.exists(arguments.output_path)
    ):
        log.warning(
            f"Output path {arguments.output_path} exists and --no-overwrite is set."
        )
        sys.exit(0)

    print_log_message_summary(arguments.level)
    try:
        processor = TextEmbeddingProcessor(arguments)
        processor.run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
