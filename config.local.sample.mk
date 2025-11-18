# Sample file for local configurations. Copy this file to config.local.mk and it will be
# included by the main Makefile.

# inform the user about this configuratio!
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
OUT_S3_BUCKET_PROCESSED_DATA := 40-processed-data-sandbox
OUT_S3_PROCESSED_VERSION := v1.0.1

# Were to write the local files
BUILD_DIR ?= build.d

# HUGGINGFACE MODEL SETTINGS
# set the model cache directory to a local project directory (default:
# ~/.cache/huggingface/transformers/)
# should be an fast local disk
HF_HOME ?= ./hf.d

# set the number of parallel jobs when processing every newspaper (each newspaper-year
# is on job; the default is 2)
MAKE_PARALLEL_OPTION ?= --jobs 2

# If you want to restrict the newspaper to work on
NEWSPAPER ?= actionfem

# suppress the logging output of make itself 
LOGGING_LEVEL := WARNING
