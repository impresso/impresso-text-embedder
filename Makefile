# Container build + Run:AI submission for impresso-text-embedder.
#
# Pure orchestration: every recipe shells out to docker / kubectl / runai.
# Settings come from .env.docker (build args, registry, Run:AI project) and
# .env (S3 + Harbor robot creds). See .progress/docker-runai/notes.md.

SHELL := /bin/bash
export SHELLOPTS := errexit:pipefail

# .env.docker holds non-secret build/runai settings (LDAP UID/GID, registry, etc).
-include .env.docker

# .env holds S3 creds (SE_*) and Harbor robot creds (HARBOR_ROBOT_*).
# Loaded here for visibility in `make help`-style targets; recipes that
# care about $-containing values shell-source it explicitly (see
# k8s-create-pull-secret).
-include .env


DOCKER_REGISTRY ?= registry.rcp.epfl.ch
DOCKER_PROJECT  ?= impresso
DOCKER_IMAGE    ?= impresso-text-embedder
DOCKER_TAG      ?= v0.1
DOCKER_IMAGE_BASE := $(DOCKER_REGISTRY)/$(DOCKER_PROJECT)/$(DOCKER_IMAGE)

# Model pin forwarded to runai-submit. Defaults mirror the Python
# DEFAULT_MODEL_REVISION in src/impresso_text_embedder/model.py so both
# paths produce byte-identical outputs. Override in .env.docker to bump.
CREATOR_NAME     ?= Alibaba-NLP
HF_MODEL_NAME    ?= gte-multilingual-base
HF_MODEL_VERSION ?= f7d567e

DOCKER_BUILD_ARGS := \
  --build-arg LDAP_UID=$(LDAP_UID) \
  --build-arg LDAP_GID=$(LDAP_GID) \
  --build-arg LDAP_USERNAME=$(LDAP_USERNAME) \
  --build-arg LDAP_GROUPNAME=$(LDAP_GROUPNAME)

docker-login:
	docker login $(DOCKER_REGISTRY)

# Build for linux/amd64 and load into the local docker daemon (Mac-friendly).
docker-build:
	docker buildx build --platform linux/amd64 --load \
	  $(DOCKER_BUILD_ARGS) \
	  -t $(DOCKER_IMAGE_BASE):$(DOCKER_TAG) \
	  .

# Build and push in one step (no local --load). Use on a Linux build host
# or when you don't need to `docker run` the image locally.
docker-build-push:
	docker buildx build --platform linux/amd64 --push \
	  $(DOCKER_BUILD_ARGS) \
	  -t $(DOCKER_IMAGE_BASE):$(DOCKER_TAG) \
	  .

docker-push:
	docker push $(DOCKER_IMAGE_BASE):$(DOCKER_TAG)

###
# KUBERNETES SECRETS

K8S_SECRET_NAME      ?= s3-credentials
K8S_PULL_SECRET_NAME ?= harbor-pull-secret

k8s-create-secrets: k8s-create-secret k8s-create-pull-secret

# S3 credentials. Idempotent (re-run when keys rotate).
k8s-create-secret:
	@test -n "$(SE_ACCESS_KEY)" || { echo "SE_ACCESS_KEY missing from .env"; exit 1; }
	@test -n "$(SE_SECRET_KEY)" || { echo "SE_SECRET_KEY missing from .env"; exit 1; }
	@test -n "$(SE_HOST_URL)"   || { echo "SE_HOST_URL missing from .env"; exit 1; }
	kubectl create secret generic $(K8S_SECRET_NAME) \
	  --from-literal=SE_ACCESS_KEY=$(SE_ACCESS_KEY) \
	  --from-literal=SE_SECRET_KEY=$(SE_SECRET_KEY) \
	  --from-literal=SE_HOST_URL=$(SE_HOST_URL) \
	  --dry-run=client -o yaml | kubectl apply -f -

# Harbor pull secret. Robot username has the form robot$$<project>+<robot> —
# `$` would be mangled by make, so we shell-source .env instead of relying
# on `make`'s -include.
k8s-create-pull-secret:
	@set -a; . ./.env; set +a; \
	test -n "$$HARBOR_ROBOT_USERNAME" || { echo "HARBOR_ROBOT_USERNAME missing from .env"; exit 1; }; \
	test -n "$$HARBOR_ROBOT_PASSWORD" || { echo "HARBOR_ROBOT_PASSWORD missing from .env"; exit 1; }; \
	kubectl create secret docker-registry $(K8S_PULL_SECRET_NAME) \
	  --docker-server=$(DOCKER_REGISTRY) \
	  --docker-username="$$HARBOR_ROBOT_USERNAME" \
	  --docker-password="$$HARBOR_ROBOT_PASSWORD" \
	  --dry-run=client -o yaml | kubectl apply -f -

###
# RUN:AI

# Silence the v1-CLI deprecation banner.
export SUPPRESS_DEPRECATION_MESSAGE := true

RUNAI_PROJECT    ?= dhlab-journe
RCP_PVC          ?= dhlab-scratch
RCP_SCRATCH_PATH ?= /rcp-scratch
RUNAI_NODE_POOL  ?= default

# Optional GPU product selector passed as `runai submit --node-type`. When
# unset, Run:AI schedules on whatever GPU the node pool offers. Set it to
# target a specific arch (e.g. RUNAI_GPU_TYPE=NVIDIA-H100-80GB). Exact
# labels vary per cluster; confirm on RCP before relying on this.
RUNAI_GPU_TYPE   ?=
RUNAI_GPU_TYPE_ARG := $(if $(RUNAI_GPU_TYPE),--node-type $(RUNAI_GPU_TYPE),)

# Per-job naming. PROVIDER must be set on the command line.
RUNAI_JOB_NAME ?= embed-$(shell echo $(PROVIDER) | tr '[:upper:]' '[:lower:]')

# Required at runtime.
INPUT_BUCKET  ?=
OUTPUT_BUCKET ?=

# Where the HF cache lives on the PVC. Change per-user.
HF_HOME_PVC ?= $(RCP_SCRATCH_PATH)/$(LDAP_USERNAME)/.cache/hf

# Anything appended after `make runai-submit ... -- --foo --bar` is forwarded
# to impresso-embed-create. Use this for --embedding-level, --batch-size, etc.
EMBED_EXTRA_ARGS ?=

# Submit a one-shot job for one provider.
# Usage: make runai-submit PROVIDER=BNL INPUT_BUCKET=22-rebuilt-final OUTPUT_BUCKET=42-processed-data-final
runai-submit:
	@test -n "$(PROVIDER)"      || { echo "PROVIDER is required";      exit 1; }
	@test -n "$(INPUT_BUCKET)"  || { echo "INPUT_BUCKET is required";  exit 1; }
	@test -n "$(OUTPUT_BUCKET)" || { echo "OUTPUT_BUCKET is required"; exit 1; }
	runai submit \
	  --name $(RUNAI_JOB_NAME) \
	  --project $(RUNAI_PROJECT) \
	  --image $(DOCKER_IMAGE_BASE):$(DOCKER_TAG) \
	  --gpu 1 \
	  --existing-pvc claimname=$(RCP_PVC),path=$(RCP_SCRATCH_PATH) \
	  --environment SE_ACCESS_KEY=SECRET:$(K8S_SECRET_NAME),SE_ACCESS_KEY \
	  --environment SE_SECRET_KEY=SECRET:$(K8S_SECRET_NAME),SE_SECRET_KEY \
	  --environment SE_HOST_URL=SECRET:$(K8S_SECRET_NAME),SE_HOST_URL \
	  --environment HF_HOME=$(HF_HOME_PVC) \
	  --node-pools $(RUNAI_NODE_POOL) \
	  $(RUNAI_GPU_TYPE_ARG) \
	  -- --provider $(PROVIDER) \
	     --input-bucket $(INPUT_BUCKET) \
	     --output-bucket $(OUTPUT_BUCKET) \
	     --model-name $(CREATOR_NAME)/$(HF_MODEL_NAME) \
	     --model-revision $(HF_MODEL_VERSION) \
	     $(EMBED_EXTRA_ARGS)

# Interactive debug pod: overrides the entrypoint with `sleep infinity` so
# you can shell in and check PVC mount, secrets, GPU, model cache, etc.
RUNAI_DEBUG_JOB_NAME ?= embed-debug
RUNAI_DEBUG_GPU      ?= 1

runai-interactive:
	runai submit \
	  --name $(RUNAI_DEBUG_JOB_NAME) \
	  --project $(RUNAI_PROJECT) \
	  --image $(DOCKER_IMAGE_BASE):$(DOCKER_TAG) \
	  --interactive \
	  --gpu $(RUNAI_DEBUG_GPU) \
	  --existing-pvc claimname=$(RCP_PVC),path=$(RCP_SCRATCH_PATH) \
	  --environment SE_ACCESS_KEY=SECRET:$(K8S_SECRET_NAME),SE_ACCESS_KEY \
	  --environment SE_SECRET_KEY=SECRET:$(K8S_SECRET_NAME),SE_SECRET_KEY \
	  --environment SE_HOST_URL=SECRET:$(K8S_SECRET_NAME),SE_HOST_URL \
	  --environment HF_HOME=$(HF_HOME_PVC) \
	  --node-pools $(RUNAI_NODE_POOL) \
	  $(RUNAI_GPU_TYPE_ARG) \
	  --command -- sleep infinity
	@echo
	@echo "Pod submitted. When Running:"
	@echo "  make runai-bash         # shell in"
	@echo "  make runai-delete-debug # remove when done"

runai-bash:
	runai bash $(RUNAI_DEBUG_JOB_NAME) -p $(RUNAI_PROJECT)

runai-delete-debug:
	runai delete job $(RUNAI_DEBUG_JOB_NAME) -p $(RUNAI_PROJECT)

###
# LOCAL UTILITIES

TMP_DIR ?= tmp

# Download an S3 object (using SE_* creds from .env) and decompress it locally.
# Handy for eyeballing one output shard on your laptop.
# Usage: make s3-fetch s3://42-processed-data-final/embeddings/docs/embeddings_gte_v1-1-0/BNL/actionfem/actionfem-1927.jsonl.bz2
#
# The URL is a positional arg. The ifeq block below grabs it out of
# MAKECMDGOALS; the %: stub stops make from trying to build the URL as a
# real goal. It only kicks in when s3-fetch is the first goal, so other
# targets keep their normal "No rule to make target …" behavior.
ifeq (s3-fetch,$(firstword $(MAKECMDGOALS)))
S3_TARGET ?= $(word 2,$(MAKECMDGOALS))
%:
	@:
endif

s3-fetch:
	@test -n "$(S3_TARGET)" || { echo "Usage: make s3-fetch s3://bucket/key"; exit 1; }
	@mkdir -p $(TMP_DIR)
	@set -a; . ./.env; set +a; \
	uv run python -c "import os, bz2, shutil; from urllib.parse import urlparse; from impresso_text_embedder.io import get_s3_client; u = urlparse('$(S3_TARGET)'); b, k = u.netloc, u.path.lstrip('/'); fn = (b + '/' + k).replace('/', '-'); p = os.path.join('$(TMP_DIR)', fn); print(f'Downloading s3://{b}/{k} -> {p}'); get_s3_client().download_file(b, k, p); out = p[:-4] if p.endswith('.bz2') else p + '.out'; print(f'Decompressing -> {out}'); shutil.copyfileobj(bz2.open(p, 'rb'), open(out, 'wb')); p.endswith('.bz2') and os.remove(p); print(f'Done. Removed {p}' if p.endswith('.bz2') else 'Done.')"

###
# HELP

help:
	@echo "Targets:"
	@echo "  Docker:"
	@echo "    docker-login         Login to $(DOCKER_REGISTRY) (Gaspar creds)"
	@echo "    docker-build         Build linux/amd64 image, load locally"
	@echo "    docker-build-push    Build and push in one step (no local load)"
	@echo "    docker-push          Push an already-built image"
	@echo "  Kubernetes secrets:"
	@echo "    k8s-create-secrets       Both: S3 + Harbor pull"
	@echo "    k8s-create-secret        S3 creds from .env"
	@echo "    k8s-create-pull-secret   Harbor pull creds from .env"
	@echo "  Run:AI:"
	@echo "    runai-submit         PROVIDER=… INPUT_BUCKET=… OUTPUT_BUCKET=… [EMBED_EXTRA_ARGS=…] [RUNAI_GPU_TYPE=…]"
	@echo "    runai-interactive    Submit a debug pod (sleep infinity) [RUNAI_GPU_TYPE=…]"
	@echo "    runai-bash           Shell into the debug pod"
	@echo "    runai-delete-debug   Delete the debug pod"
	@echo "  Local utilities:"
	@echo "    s3-fetch             s3://… [TMP_DIR=tmp]  Download + bunzip2 one object"
	@echo
	@echo "  RUNAI_GPU_TYPE pins the GPU arch (e.g. NVIDIA-H100-80GB). Leave"
	@echo "  unset to take whatever the node pool offers. Same image runs on"
	@echo "  A100 and H100; the encoder auto-detects the arch at startup."
	@echo
	@echo "  Model pin passed to runai-submit (override in .env.docker):"
	@echo "    CREATOR_NAME=$(CREATOR_NAME) HF_MODEL_NAME=$(HF_MODEL_NAME) HF_MODEL_VERSION=$(HF_MODEL_VERSION)"
	@echo
	@echo "First-time setup: cp .env.docker.example .env.docker && fill in."
	@echo "S3 + Harbor robot creds go in .env (gitignored)."

.DEFAULT_GOAL := help

.PHONY: help \
        docker-login docker-build docker-build-push docker-push \
        k8s-create-secrets k8s-create-secret k8s-create-pull-secret \
        runai-submit runai-interactive runai-bash runai-delete-debug \
        s3-fetch
