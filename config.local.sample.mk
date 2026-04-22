# Sample file for local configurations. Copy this file to config.local.mk and it will be
# included by the main Makefile.

# inform the user about this configuration!
$(info Make: Including config.local.mk: $(shell readlink -f config.local.mk))

# Typical adaptations
# if NAME is already set to a non-empty value, it will not be overwritten!
# NAME ?= VALUE
# NAME will be set to this VALUE, even if it was already set!
# NAME := VALUE
# Warning: Never add a trailing # comment after a variable assignment, it will break the
# code!
# NAME := VALUE # THIS WILL BREAK THE CODE!

# The s3 output path is computed from 3 make variables.
# You can set two of them here! OUT_S3_BUCKET_PROCESSED_DATA and OUT_S3_PROCESSED_VERSION
# Don't change the OUT_S3_PROCESSED_INFIX!
# : s3://$OUT_S3_BUCKET_PROCESSED_DATA/$OUT_S3_PROCESSED_INFIX/$OUT_S3_PROCESSED_VERSION

# ----------------------------------------------------------------
# User-specific settings start here
# ----------------------------------------------------------------

# Were to write the local files
BUILD_DIR := build.d

# Set the model name and version
CREATOR_NAME := Alibaba-NLP
HF_MODEL_NAME := gte-multilingual-base
HF_MODEL_VERSION := f7d567e
HF_FULL_MODEL_NAME := $(CREATOR_NAME)/$(HF_MODEL_NAME)
  $(call log.debug, HF_FULL_MODEL_NAME)

# The embedding level option: text, sentence or chunk
EMBEDDING_LEVEL_OPTION := chunk

# The input bucket
IN_S3_PREFIXES := lingproc/lingproc-test-v1.0.0
IN_S3_BUCKET_REBUILT := 000-processing-test-samples

OUT_S3_BUCKET_PROCESSED_DATA := 140-processed-data-sandbox

# The output infix and version see internal documentation for more file structure information
OUT_S3_PROCESSED_INFIX := textembeddings-$(HF_MODEL_NAME)
OUT_S3_PROCESSED_VERSION := v1.0.1


# Set the minimum character length for the text to be included for embedding.
# Texts shorter than this length will not be embedded and will be skipped entirely (not showing up in the output at all).
# Set to 10 by default because for sentences 800 characters is too long and 5 is too short.
EMBEDDING_MIN_CHAR_LENGTH := 10
  $(call log.debug, EMBEDDING_MIN_CHAR_LENGTH)

# HUGGINGFACE MODEL SETTINGS
# set the model cache directory to a local project directory (default:
# ~/.cache/huggingface/transformers/)
# should be a fast local disk
HF_HOME := ./hf.d

# set the number of parallel jobs when processing every newspaper (each newspaper-year
# is on job; the default is 2)
MAKE_PARALLEL_OPTION := --jobs 2

# If you want to restrict the newspaper to work on
# Provider + Newspaper (pair)
PROVIDER  := SNL
NEWSPAPER := EXP

# suppress the logging output of make itself
LOGGING_LEVEL := WARNING