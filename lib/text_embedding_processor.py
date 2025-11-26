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
import sys
import datetime
from tqdm import tqdm
from collections import Counter
import time
from smart_open import open
from typing import Any, Dict, Iterator, Tuple, Generator, List
from sentence_transformers import SentenceTransformer
import torch
import boto3
from dotenv import load_dotenv
from utils import print_log_message_summary

random.seed(42)
log = logging.getLogger(__name__)
JSONType = Dict[str, Any]


def rebuild_ft_from_offsets(sents):
    """
    Reconstructs the full text from a list of sentence dictionaries using character offsets.

    Args:
        sents (List[Dict]): List of sentence dictionaries, each containing a "tok" key
            with a list of token dictionaries. Each token dictionary should have:
                - "t": token text (str)
                - "o": character offset (int)

    Returns:
        str: The reconstructed text as a single string.
    """
    # Flatten all tokens across sentences
    toks = []
    for sent in sents:
        toks.extend(sent.get("tok", []))

    if not toks:
        return ""

    # Sort tokens by offset
    toks = sorted(toks, key=lambda x: x.get("o", 0))

    text = []
    current_pos = 0

    for tok in toks:
        token_text = tok["t"]
        offset = tok["o"]

        # Add missing spaces or characters between tokens
        if offset > current_pos:
            text.append(" " * (offset - current_pos))

        # Insert the token
        text.append(token_text)

        # Move cursor
        current_pos = offset + len(token_text)

    return "".join(text)


def rebuild_sentence_from_offsets(sent):
    toks = sorted(sent.get("tok", []), key=lambda x: x["o"])

    if not toks:
        return ""
    text = []
    current_pos = toks[0]["o"]

    for tok in toks:
        offset = tok["o"]
        token_text = tok["t"]

        if offset > current_pos:
            text.append(" " * (offset - current_pos))

        text.append(token_text)
        current_pos = offset + len(token_text)

    return "".join(text).strip()


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
        m = SentenceTransformer(model_name_or_path=self.args.model_name,
                                trust_remote_code=True,
                                revision=self.args.model_revision
                                )

        log.info("Model loaded.")
        # Check the device of the model's parameters
        if next(m.parameters()).is_cuda:  # Check if the model is on a CUDA device
            current_device = next(m.parameters()).device
            device_name = torch.cuda.get_device_name(current_device)
            log.info(f"Model loaded on GPU: {current_device} ({device_name})")
        else:
            log.info("Model loaded on CPU.")
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

    def compute_embeddings(self, data: JSONType) -> JSONType | List[JSONType] | None:
        """Computes embeddings at the text, sentence, or chunk level depending on args or data structure.

        Returns:
            - For article/text embeddings: List[JSONType] (one or more flat records)
            - For sentence embeddings: JSONType matching embeddings-sentence.schema.json
            - For chunk embeddings: JSONType matching embeddings-chunks.schema.json
            - None if nothing to embed (too short, filtered type, etc.)
        """

        embedder = self.args.model_name + "@" + (self.args.model_revision or "default")

        # Content type filter (tp is content item type)
        content_item_type = data.get("tp")
        if content_item_type and content_item_type not in self.args.content_type:
            self.stats[f"skipped_type_{content_item_type}"] += 1
            return None

        has_sentences = isinstance(data.get("sents"), list)

        # Load model lazily
        if self.model is None:
            self.model = self.load_model()

        ci_id = data.get("id")  # canonical content item id
        lang = data.get("lg")  # if present; optional in schemas

        # --------------------------------------------------------------------------------
        # HELPER: timestamp in schema format: YYYY-MM-DDTHH:MM:SSZ
        # --------------------------------------------------------------------------------
        self.last_timestamp = datetime.datetime.utcnow().replace(microsecond=0)
        ts_str = self.last_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        # --------------------------------------------------------------------------------
        # 1) CHUNK-LEVEL EMBEDDINGS  (embeddings-chunks.schema.json)
        # --------------------------------------------------------------------------------

        # TODO: to move to lingproc
        # text = data.get("ft", "")
        sents = data.get("sents", [])
        text = rebuild_ft_from_offsets(sents)
        textlen = len(text)
        log.info(
            f"Computing embedding for ID: {ci_id}, text length: {textlen}, at the {self.args.embedding_level} level")

        if self.args.embedding_level == "chunk":
            if not text or textlen <= self.args.min_char_length:
                self.stats["short_texts"] += 1
                return None

            from chonkie import SemanticChunker

            chunker = SemanticChunker(
                embedding_model="minishlab/potion-base-8M",
                threshold=0.5,  # Similarity threshold
                chunk_size=1024,  # Maximum tokens per chunk
                min_sentences=5  # Initial sentences per chunk
            )

            chunks = chunker.chunk(text)
            if not chunks:
                return None

            # Prepare texts for encoding
            chunk_texts = [chunk.text for chunk in chunks]

            start_time = time.time()
            chunk_embeddings = self.model.encode(
                chunk_texts,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.args.normalize_embeddings,
            )
            end_time = time.time()

            self.stats["total_time"] += end_time - start_time
            self.stats["valid_texts"] += len(chunk_texts)

            # Build schema-compliant structure
            chunk_items = []
            for idx, (chunk, emb_vec) in enumerate(
                    tqdm(zip(chunks, chunk_embeddings), total=len(chunks))
            ):
                emb_list = [round(n, 5) for n in emb_vec.tolist()]
                size = len(emb_list)

                # Try to get character offset from chonkie, but it's optional in schema
                offset = (
                        getattr(chunk, "start", None)
                        or getattr(chunk, "start_char", None)
                        or None
                )

                item: JSONType = {
                    "chunk_id": idx,
                    "embedding": emb_list,
                    "size": size,
                }
                if lang:
                    item["lg"] = lang
                if offset is not None:
                    item["o"] = offset

                chunk_items.append(item)

            result_doc: JSONType = {
                "ts": ts_str,
                "ci_id": ci_id,
                "chunks": chunk_items,
            }

            if hasattr(self.args, "model_id") and self.args.model_id:
                result_doc["model_id"] = self.args.model_id
            if "lingproc_path" in data:
                result_doc["lingproc_path"] = data["lingproc_path"]
            if hasattr(self.args, "git_commit") and self.args.git_commit:
                result_doc["git"] = self.args.git_commit

            log.debug(f"Computed {len(chunk_items)} chunk embeddings for CI: {ci_id}")
            return result_doc

        # --------------------------------------------------------------------------------
        # 2) SENTENCE-LEVEL EMBEDDINGS  (embeddings-sentence.schema.json)
        #     Also used when embedding_level == "text" but the document already has sents.
        # --------------------------------------------------------------------------------
        elif self.args.embedding_level == "sentence" and not has_sentences:
            log.warning(
                f"Sentence-level embedding requested but no sentences found for CI: "
                f"{ci_id}"
            )
            return None

        elif self.args.embedding_level == "sentence" and has_sentences:
            sents = data.get("sents", [])
            if not sents:
                return None

            texts_to_embed: list[tuple[int, str, int | None]] = []  # (sent_id, text, offset)

            for s_idx, s in tqdm(enumerate(sents), total=len(sents)):
                # Build sentence text from tokens
                sent_text = rebuild_sentence_from_offsets(s)
                sent_text = sent_text.strip()

                if len(sent_text) > self.args.min_char_length:
                    # sentence-level offset (optional in schema)
                    offset = s.get("o") if isinstance(s, dict) else None
                    texts_to_embed.append((s_idx, sent_text, offset))
                else:
                    self.stats["short_texts"] += 1

            if not texts_to_embed:
                return None

            sent_texts = [txt for _, txt, _ in texts_to_embed]

            start_time = time.time()
            sent_embeddings = self.model.encode(
                sent_texts,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.args.normalize_embeddings,
            )
            end_time = time.time()

            self.stats["total_time"] += end_time - start_time
            self.stats["valid_texts"] += len(sent_texts)

            sent_items = []
            for (sent_id, _txt, offset), emb_vec in tqdm(
                    zip(texts_to_embed, sent_embeddings),
                    total=len(texts_to_embed),
                    desc="Sentence embeddings"
            ):
                emb_list = [round(n, 5) for n in emb_vec.tolist()]
                size = len(emb_list)

                item: JSONType = {
                    "sent_id": sent_id,
                    "embedding": emb_list,
                    "size": size,
                }
                if lang:
                    item["lg"] = lang
                if offset is not None:
                    item["o"] = offset

                sent_items.append(item)

            result_doc: JSONType = {
                "ts": ts_str,
                "ci_id": ci_id,
                "sents": sent_items,
            }

            # Optional metadata fields, if available:
            # if hasattr(self.args, "model_id") and self.args.model_id:
            #     result_doc["model_id"] = self.args.model_id
            # if "lingproc_path" in data:
            #     result_doc["lingproc_path"] = data["lingproc_path"]
            # if hasattr(self.args, "git_commit") and self.args.git_commit:
            #     result_doc["git"] = self.args.git_commit

            log.debug(f"Computed {len(sent_items)} sentence embeddings for CI: {ci_id}")
            return result_doc

        # --------------------------------------------------------------------------------
        # 3) TEXT / ARTICLE-LEVEL EMBEDDINGS  (flat records, as you had before)
        # --------------------------------------------------------------------------------
        elif self.args.embedding_level == "text":
            if not text or textlen <= self.args.min_char_length:
                log.info("Skipping text embedding due to too short text")
                self.stats["short_texts"] += 1
                return None

            texts = [text]

            start_time = time.time()
            doc_embeddings = self.model.encode(
                texts,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.args.normalize_embeddings,
            )
            end_time = time.time()

            self.stats["total_time"] += end_time - start_time
            self.stats["valid_texts"] += len(texts)

            results: list[JSONType] = []
            for emb_vec in doc_embeddings:
                emb_list = [round(n, 5) for n in emb_vec.tolist()]
                result: JSONType = {
                    "id": ci_id,
                    "ts": ts_str,
                    "embedder": embedder,
                    "len": len(text),
                    "embedding": emb_list,
                }
                if self.args.include_text:
                    result["text"] = text
                results.append(result)

            log.debug(f"Computed {len(results)} document embeddings for ID: {ci_id}")
            return results

        # Fallback (should not be reached)
        return None

    def write_embeddings(self, embeddings: Iterator[JSONType]) -> None:
        """Writes computed embeddings to the output file in JSON format.

        Supports:
          - List[dict] (article embeddings: one JSON line per embedding)
          - dict (sentence/chunk schemas: one JSON line per content item)
        """
        output_file_path = self.args.output_path
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for embedding in embeddings:
                if not embedding:
                    continue

                # Case 1: article-level: list of flat records
                if isinstance(embedding, list):
                    for rec in embedding:
                        if not rec:
                            continue
                        log.debug(
                            "Writing article embedding: %s",
                            rec.get("id") or rec.get("ci_id"),
                        )
                        outfile.write(
                            json.dumps(
                                rec,
                                indent=None,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        self.stats["files_created"] += 1

                # Case 2: sentence/chunk-level: top-level dict matching schema
                elif isinstance(embedding, dict):
                    log.debug(
                        "Writing structured embedding for CI: %s",
                        embedding.get("ci_id") or embedding.get("id"),
                    )
                    outfile.write(
                        json.dumps(
                            embedding,
                            # indent=None, # compact output
                            # separators=(",", ":"), # keep spaces for readability
                            ensure_ascii=True,  # ASCII set to TRUE for compatibility and readability
                        )
                        + "\n"
                    )
                    self.stats["files_created"] += 1

                # Unexpected type
                else:
                    log.warning(
                        "Unexpected embedding object of type %s, skipping.",
                        type(embedding),
                    )

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
        "--embedding-level",
        choices=["text", "sentence", "chunk"],
        default="text",
        help=(
            "Specify the embedding level: "
            "'text' (default, one embedding per document), "
            "'sentence' (one embedding per sentence), or "
            "'chunk' (chunk text using chonkie)."
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
