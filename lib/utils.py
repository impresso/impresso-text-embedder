import logging

log = logging.getLogger(__name__)


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
