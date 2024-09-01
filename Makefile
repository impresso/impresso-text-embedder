# Define shell to use and enable strict error handling
SHELL:=/bin/bash
export SHELLOPTS:=errexit:pipefail
.SECONDARY:
.DELETE_ON_ERROR:

# Default target when no target is specified
.DEFAULT_GOAL := help

# Include the logging functions and set the logging level: DEBUG, INFO, WARNING, ERROR
LOGGING_LEVEL ?= INFO
include lib/log.mk

# Define default variables
# A file containing a space-separated line with all newspapers to process
# Feel free to handcraft another file with the newspapers you want to process
# This file is automatically populated from the content of s3 rebuilt bucket
NEWSPAPER_LIST_FILE ?= $(BUILD_DIR)/newspapers.txt

# Huggingface model variables
CREATOR_NAME ?= Alibaba-NLP
HF_MODEL_NAME ?= gte-multilingual-base
HF_MODEL_VERSION ?= f7d567e
HF_FULL_MODEL_NAME ?= $(CREATOR_NAME)/$(HF_MODEL_NAME)

# Further OPTIONS
#EMBEDDING_INCLUDE_TEXT_OPTION ?= --include-text
EMBEDDING_INCLUDE_TEXT_OPTION ?= 

EMBEDDING_MIN_CHAR_LENGTH ?= 800

# UPLOAD OPTION
# Decomment next line to enable dry-run mode for s3 output (independently of the
# --s3-output-path setting)
# EMBEDDING_S3_OUTPUT_DRY_RUN?= --s3-output-dry-run 
EMBEDDING_S3_OUTPUT_DRY_RUN ?= 
  $(call log.info, EMBEDDING_S3_OUTPUT_DRY_RUN)

# Keep only the local timestam output files after uploading (only relevant when
# uploading to s3)

EMBEDDING_KEEP_TIMESTAMP_ONLY_OPTION ?= --keep-timestamp-only
#EMBEDDING_KEEP_TIMESTAMP_ONLY_OPTION ?= 

# double check if the output file exists in s3 and quit if it does
EMBEDDING_QUIT_IF_S3_OUTPUT_EXISTS ?= --quit-if-s3-output-exists

MAKE_PARALLEL_OPTION ?= --jobs 4


# set the model cache directory to a local project directory (default: ~/.cache/huggingface/transformers/)
HF_HOME?=./hf.d

BUILD_DIR ?= build.d

# the newspaper to process, can also be actionfem/actionfem-1933  ... just a prefix into
# s3 is ok!
NEWSPAPER_FILTER ?= actionfem

# The input bucket
IN_S3_BUCKET ?= 22-rebuilt-final

# The input path
IN_S3_PATH := s3://$(IN_S3_BUCKET)/$(NEWSPAPER_FILTER)
  $(call log.info, IN_S3_PATH)

# The local path
IN_LOCAL_PATH := $(BUILD_DIR)/$(IN_S3_BUCKET)/$(NEWSPAPER_FILTER)
  $(call log.info, IN_LOCAL_PATH)

# The output bucket
OUT_S3_BUCKET ?= 42-processed-data-final
OUT_S3_PROCESSED_INFIX ?= textembeddings-$(HF_MODEL_NAME)
OUT_S3_PROCESSED_VERSION ?= v1.0.0

# The output path
OUT_S3_PATH := s3://$(OUT_S3_BUCKET)/$(OUT_S3_PROCESSED_INFIX)/$(OUT_S3_PROCESSED_VERSION)/$(NEWSPAPER_FILTER)
  $(call log.info, OUT_S3_PATH)

# The local path
OUT_LOCAL_PATH := $(BUILD_DIR)/$(OUT_S3_BUCKET)/$(OUT_S3_PROCESSED_INFIX)/$(OUT_S3_PROCESSED_VERSION)/$(NEWSPAPER_FILTER)
  $(call log.info, OUT_LOCAL_PATH)


help:
	@echo "Usage: make [target] [-j <jobs>]"
	@echo ""
	@echo "Targets:"
	@echo "  setup: Create the local directories and setup the HF model"
	@echo "  sync: Sync the data from the S3 bucket to the local directory"
	@echo "  resync: Remove the local synchronization file stamp and redoes everything, ensuring a full sync with the remote server."
	@echo "  newspaper: Process the text embeddings for the given newspaper"
	@echo "  help: Show this help message"


.PHONY: all  help


newspaper: textembedding-target


each:
	for NEWSPAPER_FILTER in $(file < $(NEWSPAPER_LIST_FILE)) ; do \
		$(MAKE) NEWSPAPER_FILTER=$$NEWSPAPER_FILTER resync  ; \
		$(MAKE) $(MAKE_PARALLEL_OPTION) NEWSPAPER_FILTER=$$NEWSPAPER_FILTER newspaper  ; \
	done

setup:
	pipenv install
	# Create the local directory
	mkdir -p $(IN_LOCAL_PATH)
	mkdir -p $(OUT_LOCAL_PATH)
	$(MAKE) setup-hf-model

setup-hf-model:
	pipenv run python -c "from sentence_transformers import SentenceTransformer as st; m = st('$(HF_FULL_MODEL_NAME)', revision='$(HF_MODEL_VERSION)',trust_remote_code=True); len(m.encode('This is a test!')) or exit(1)"


# IN_S3_PATH last synced stamps

last-synced-stamp-files += $(BUILD_DIR)/$(IN_S3_BUCKET)/$(NEWSPAPER_FILTER).last_synced

last-synced-output-files += $(OUT_LOCAL_PATH).last_synced
  $(call log.info, last-synced-output-files)
sync: sync-input sync-output
sync-input: $(last-synced-stamp-files) 
sync-output: $(last-synced-output-files)

# Remove the local synchronization file stamp and redoes everything, ensuring a full sync with the remote server.
resync: clean-sync
	$(MAKE) sync

clean-sync:
	rm -f $(last-synced-stamp-files) $(last-synced-output-files) || true


newspaper-target: $(BUILD_DIR)/newspapers.txt

$(BUILD_DIR)/newspapers.txt:
	python -c \
	"import lib.s3_to_local_stamps as m; \
	s3 = m.get_s3_resource(); \
	bucket = s3.Bucket('$(IN_S3_BUCKET)'); \
    result = bucket.meta.client.list_objects_v2(Bucket=bucket.name, Delimiter='/'); \
    print(*(prefix['Prefix'][:-1] for prefix in result.get('CommonPrefixes', [])))" \
	> $@

$(BUILD_DIR)/$(IN_S3_BUCKET)/$(NEWSPAPER_FILTER).last_synced:
	python lib/s3_to_local_stamps.py \
	   $(IN_S3_PATH) \
	   --local-dir $(BUILD_DIR) \
	   --stamp-extension .stamp \
	   2> >(tee $@.log >&2) && \
	touch $@

$(OUT_LOCAL_PATH).last_synced:
	python lib/s3_to_local_stamps.py \
	   $(OUT_S3_PATH) \
	   --local-dir $(BUILD_DIR) \
	   --stamp-extension '' \
	   2> >(tee $@.log >&2) && \
	touch $@



# variable for all locally available rebuilt stamp files. Needed for dependency tracking
# of the build process
local-rebuilt-stamp-files := $(shell ls -r $(IN_LOCAL_PATH)/*.jsonl.bz2.stamp)
  $(call log.info, local-rebuilt-stamp-files)

define local_rebuilt_stamp_to_local_textembedding_file
$(1:$(IN_LOCAL_PATH)/%.jsonl.bz2.stamp=$(OUT_LOCAL_PATH)/%.jsonl.bz2)
endef

local-textembedding-files:=\
 $(call local_rebuilt_stamp_to_local_textembedding_file,$(local-rebuilt-stamp-files))

  $(call log.info, local-textembedding-files)

textembedding-target: $(local-textembedding-files)

# Rule for the langdata files 
# --language-file-s3-path  $(<:$(BUILD_DIR)/%=s3://%) \ if reading from s3
$(OUT_LOCAL_PATH)/%.jsonl.bz2: $(IN_LOCAL_PATH)/%.jsonl.bz2.stamp
	mkdir -p $(@D)
	python lib/text_embedding_processor.py \
	  --min-char-length $(EMBEDDING_MIN_CHAR_LENGTH) \
	  $(EMBEDDING_INCLUDE_TEXT_OPTION) \
	  --model-name $(HF_FULL_MODEL_NAME) \
	  --model-revision $(HF_MODEL_VERSION) \
	  --input-path $(call local_to_s3,$<,.stamp) \
	  --output-path $@ \
	  --s3-output-path $(call local_to_s3,$@) \
	  $(EMBEDDING_S3_OUTPUT_DRY_RUN) \
	  $(EMBEDDING_QUIT_IF_S3_OUTPUT_EXISTS) \
	  $(EMBEDDING_KEEP_TIMESTAMP_ONLY_OPTION) \
	  2> >( tee $@.log >&2 )



# function to turn a local file path into a s3 file path, optionall cutting off the
# suffix given as argument
define local_to_s3
$(subst $(2),,$(subst $(BUILD_DIR),s3:/,$(1)))
endef
# Doctests for local_to_s3 function

# Example 1: Convert local path to S3 path without stripping any suffix
# Input: $(call local_to_s3,build.d/22-rebuilt-final/marieclaire/file.txt)
# Output: s3://22-rebuilt-final/marieclaire/file.txt
# $(call log.debug, $(call local_to_s3,build.d/22-rebuilt-final/marieclaire/file.txt))
# Example 2: Convert local path to S3 path and strip the .txt suffix
# Input: $(call local_to_s3,build.d/22-rebuilt-final/marieclaire/file.txt,.txt)
# Output: s3://22-rebuilt-final/marieclaire/file

# Example 3: Convert local path to S3 path and strip a custom suffix
# Input: $(call local_to_s3,build.d/22-rebuilt-final/marieclaire/file.custom,.custom)
# Output: s3://22-rebuilt-final/marieclaire/file

langdata-target: $(local-langdata-files)
