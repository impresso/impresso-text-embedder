# Container build + Run:AI orchestration for the chunking-eval research
# branch. Pure shell-out wrapper around docker / kubectl / runai. Settings
# come from .env.docker (build args, registry, Run:AI project) and .env
# (S3 + Harbor robot creds).
#
# Branch goal: run one runai job per scenario from
# `impresso_text_embedder.research.scenarios`. The provider-walking
# `runai-submit` flow that lived here on `main` is gone — this branch
# never embeds the full Impresso corpus; it only sweeps the small
# research shard built by `corpus-fetch`.

SHELL := /bin/bash
export SHELLOPTS := errexit:pipefail

-include .env.docker
-include .env


DOCKER_REGISTRY ?= registry.rcp.epfl.ch
DOCKER_PROJECT  ?= impresso
DOCKER_IMAGE    ?= impresso-text-embedder
# Image tag = current git branch so research-branch images don't collide
# with main's. Slashes are not legal in docker tags, so `research/chunking-eval`
# becomes `research-chunking-eval`. Override via .env.docker if needed.
GIT_BRANCH      := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-')
DOCKER_TAG      ?= $(GIT_BRANCH)
DOCKER_IMAGE_BASE := $(DOCKER_REGISTRY)/$(DOCKER_PROJECT)/$(DOCKER_IMAGE)

# Model pin forwarded to every scenario job. Defaults mirror
# DEFAULT_MODEL_REVISION in src/impresso_text_embedder/model.py so the
# scenario runs and the production embedder produce comparable outputs.
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

docker-build:
	docker buildx build --platform linux/amd64 --load \
	  $(DOCKER_BUILD_ARGS) \
	  -t $(DOCKER_IMAGE_BASE):$(DOCKER_TAG) \
	  .

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

k8s-create-secret:
	@test -n "$(SE_ACCESS_KEY)" || { echo "SE_ACCESS_KEY missing from .env"; exit 1; }
	@test -n "$(SE_SECRET_KEY)" || { echo "SE_SECRET_KEY missing from .env"; exit 1; }
	@test -n "$(SE_HOST_URL)"   || { echo "SE_HOST_URL missing from .env"; exit 1; }
	kubectl create secret generic $(K8S_SECRET_NAME) \
	  --from-literal=SE_ACCESS_KEY=$(SE_ACCESS_KEY) \
	  --from-literal=SE_SECRET_KEY=$(SE_SECRET_KEY) \
	  --from-literal=SE_HOST_URL=$(SE_HOST_URL) \
	  --dry-run=client -o yaml | kubectl apply -f -

# Harbor pull secret. Robot username has the form robot$$<project>+<robot>;
# `$` would be mangled by make, so shell-source .env instead of relying
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
# RUN:AI — chunking-eval scenario sweep

export SUPPRESS_DEPRECATION_MESSAGE := true

RUNAI_PROJECT    ?= dhlab-journe
RCP_PVC          ?= dhlab-scratch
RCP_SCRATCH_PATH ?= /rcp-scratch
RUNAI_NODE_POOL  ?= default

# Optional GPU product selector passed as `runai submit --node-type`.
# Same image runs on A100 / H100 / H200; accel.py auto-detects the arch.
# Find labels with: kubectl get nodes -L nvidia.com/gpu.product
# Examples: NVIDIA-H100-80GB-HBM3, NVIDIA-H200-141GB.
RUNAI_GPU_TYPE   ?=
RUNAI_GPU_TYPE_ARG := $(if $(RUNAI_GPU_TYPE),--node-type $(RUNAI_GPU_TYPE),)

RUNAI_CPU            ?=
RUNAI_CPU_LIMIT      ?=
RUNAI_MEMORY         ?=
RUNAI_MEMORY_LIMIT   ?=
RUNAI_CPU_ARG          := $(if $(RUNAI_CPU),--cpu $(RUNAI_CPU),)
RUNAI_CPU_LIMIT_ARG    := $(if $(RUNAI_CPU_LIMIT),--cpu-limit $(RUNAI_CPU_LIMIT),)
RUNAI_MEMORY_ARG       := $(if $(RUNAI_MEMORY),--memory $(RUNAI_MEMORY),)
RUNAI_MEMORY_LIMIT_ARG := $(if $(RUNAI_MEMORY_LIMIT),--memory-limit $(RUNAI_MEMORY_LIMIT),)
RUNAI_RESOURCE_ARGS    := $(RUNAI_CPU_ARG) $(RUNAI_CPU_LIMIT_ARG) $(RUNAI_MEMORY_ARG) $(RUNAI_MEMORY_LIMIT_ARG)

# Where the HF cache lives on the PVC. Change per-user.
HF_HOME_PVC ?= $(RCP_SCRATCH_PATH)/$(LDAP_USERNAME)/.cache/hf

# Active study YAML — single source of truth for all research-pipeline
# parameters. Override on the command line: `make STUDY=study-A-fit ...`
STUDY        ?= study-v1
STUDY_CONFIG ?= configs/research/$(STUDY).yaml

# Anything appended via SWEEP_EXTRA_ARGS is forwarded to
# impresso-research-embed-sweep. Use for --batch-size, --precision, etc.
SWEEP_EXTRA_ARGS ?=

# Anything appended via QUERY_EMBED_ARGS is forwarded to
# impresso-research-query-embed (e.g. --batch-size, --no-upload, --limit).
QUERY_EMBED_ARGS ?=

# Forwarded to the AIaaS query-generate CLI
# (impresso-research-query-generate). Empty by default.
QUERY_GEN_ARGS ?=

# Forwarded to the CaaS query-generate CLI
# (impresso-research-query-generate-local). Use for --attention,
# --dtype, --max-retries, --no-upload, --limit.
QUERY_GEN_LOCAL_ARGS ?=

# Per-job name for the query-embed runai job. One job per study; no
# scenario fan-out (queries are short plain strings, no chunking).
RUNAI_QUERY_EMBED_JOB_NAME ?= $(shell echo query-embed-$(STUDY) | tr '[:upper:]' '[:lower:]')

# Per-job name for the CaaS (local-LLM) query-generate runai job.
# Mirrors the query-embed naming so concurrent jobs across studies
# don't collide.
RUNAI_QUERY_GENERATE_JOB_NAME ?= $(shell echo query-generate-$(STUDY) | tr '[:upper:]' '[:lower:]')

# Single source of truth for the scenario list. Derived from the study
# config at evaluation time so the Makefile stays in sync with the YAML.
SCENARIOS := $(shell uv run --quiet python -m impresso_text_embedder.research.scenario_builder --config $(STUDY_CONFIG) --list-ids 2>/dev/null)

# Per-job naming. SCENARIO must be set on the command line.
RUNAI_JOB_NAME ?= $(shell echo embed-sweep-$(STUDY)-$(SCENARIO) | tr '[:upper:]' '[:lower:]')

# List the scenario registry for the active study (uses the Python CLI; no GPU needed).
research-list-scenarios:
	uv run impresso-research-embed-sweep --config $(STUDY_CONFIG) --list

# Submit one scenario as a runai job.
# Usage: make runai-submit-research SCENARIO=S3 [STUDY=study-A-fit] [SWEEP_EXTRA_ARGS="--batch-size 64"]
runai-submit-research:
	@test -n "$(SCENARIO)" || { echo "SCENARIO is required (e.g. SCENARIO=S3); see 'make research-list-scenarios'"; exit 1; }
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
	  $(RUNAI_RESOURCE_ARGS) \
	  --command -- impresso-research-embed-sweep \
	     --config $(STUDY_CONFIG) \
	     --scenario $(SCENARIO) \
	     $(SWEEP_EXTRA_ARGS)

# Submit every scenario as an independent runai job. Each job runs on
# one GPU; the cluster scheduler decides node placement. Re-running is
# idempotent at the S3 layer because each job writes to its own
# scenario-specific output prefix.
# Usage: make runai-submit-research-all [SWEEP_EXTRA_ARGS="--batch-size 64"]
runai-submit-research-all:
	@test -n "$(SCENARIOS)" || { echo "scenario list is empty — study config $(STUDY_CONFIG) failed to load?"; exit 1; }
	@for s in $(SCENARIOS); do \
	    echo "==> Submitting scenario $$s (study=$(STUDY))"; \
	    $(MAKE) runai-submit-research SCENARIO=$$s STUDY=$(STUDY) SWEEP_EXTRA_ARGS='$(SWEEP_EXTRA_ARGS)'; \
	done

# Submit the per-study query-embed job — encodes every query in
# <study>/queries.jsonl.bz2 once (no chunking, no aggregation, one
# vector per query) and writes <study>/queries-embedded.jsonl.bz2.
# Same image, same pinned model, same PVC + secrets as the sweep
# jobs; only the entry-point CLI differs.
# Usage: make runai-submit-query-embed [STUDY=study-A-fit] [QUERY_EMBED_ARGS="--limit 64"]
runai-submit-query-embed:
	runai submit \
	  --name $(RUNAI_QUERY_EMBED_JOB_NAME) \
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
	  $(RUNAI_RESOURCE_ARGS) \
	  --command -- impresso-research-query-embed \
	     --config $(STUDY_CONFIG) \
	     $(QUERY_EMBED_ARGS)

# Submit the per-study CaaS query-generate job — runs the LLM
# locally (transformers + bf16 + sdpa) inside the same image instead
# of calling the EPFL RCP AIaaS endpoint. Same study YAML, same
# prompts, same output schema as `make research-query-generate`;
# the only difference is *where the model lives*. Useful when AIaaS
# is throttled / unavailable, or when a study needs a fully
# self-contained run with no shared-infra coupling.
# Usage: make runai-submit-query-generate-local [STUDY=study-A-fit] [QUERY_GEN_LOCAL_ARGS="--limit 64"]
runai-submit-query-generate-local:
	runai submit \
	  --name $(RUNAI_QUERY_GENERATE_JOB_NAME) \
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
	  $(RUNAI_RESOURCE_ARGS) \
	  --command -- impresso-research-query-generate-local \
	     --config $(STUDY_CONFIG) \
	     $(QUERY_GEN_LOCAL_ARGS)

# Interactive debug pod. Same image as the sweep jobs.
RUNAI_DEBUG_JOB_NAME ?= embed-sweep-debug
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
	  $(RUNAI_RESOURCE_ARGS) \
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
# CORPUS PREP — local, run before any runai-submit-research

# Run on a laptop / login node; produces the corpus shard the sweep
# consumes. See src/impresso_text_embedder/research/corpus_select.py and
# corpus_fetch.py.
research-corpus-select:
	uv run impresso-research-corpus-select --config $(STUDY_CONFIG) $(CORPUS_SELECT_ARGS)

research-corpus-fetch:
	uv run impresso-research-corpus-fetch --config $(STUDY_CONFIG) $(CORPUS_FETCH_ARGS)

# Step 4 — synthetic query generation via EPFL RCP AIaaS.
# Reads RCP_API_KEY from .env (via python-dotenv inside the CLI); set
# QUERY_GEN_ARGS to forward extra flags (e.g. QUERY_GEN_ARGS="--limit 1").
research-query-generate:
	uv run impresso-research-query-generate --config $(STUDY_CONFIG) $(QUERY_GEN_ARGS)

# Step 4 — CaaS variant: same prompts/schema, LLM loaded locally via
# transformers (bf16 + sdpa) instead of the AIaaS endpoint. Needs a
# GPU + the model weights on disk; on a laptop use this only with
# QUERY_GEN_LOCAL_ARGS="--limit 1 --no-upload" for a smoke run.
research-query-generate-local:
	uv run impresso-research-query-generate-local --config $(STUDY_CONFIG) $(QUERY_GEN_LOCAL_ARGS)

# Step 7 — local invocation of query-embed (no Run:AI). Encodes
# every query in the queries shard once. Useful for laptop smoke
# runs (`QUERY_EMBED_ARGS="--no-upload --limit 16 --log-dir tmp"`).
research-query-embed:
	uv run impresso-research-query-embed --config $(STUDY_CONFIG) $(QUERY_EMBED_ARGS)

###
# LOCAL UTILITIES

TMP_DIR ?= tmp

# Download an S3 object using SE_* creds and decompress it locally.
# Usage: make s3-fetch s3://140-processed-data-sandbox/chunking-eval/A-fit/S3.jsonl.bz2
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
	@echo "Targets (chunking-eval research branch):"
	@echo "  Docker:"
	@echo "    docker-login            Login to $(DOCKER_REGISTRY) (Gaspar creds)"
	@echo "    docker-build            Build linux/amd64 image, load locally"
	@echo "    docker-build-push       Build and push in one step"
	@echo "    docker-push             Push an already-built image"
	@echo "  Kubernetes secrets:"
	@echo "    k8s-create-secrets      Both: S3 + Harbor pull"
	@echo "    k8s-create-secret       S3 creds from .env"
	@echo "    k8s-create-pull-secret  Harbor pull creds from .env"
	@echo "  Corpus prep (run locally before submitting):"
	@echo "    research-corpus-select        Build the per-language manifest"
	@echo "    research-corpus-fetch         Materialise manifest -> corpus shard on S3"
	@echo "    research-query-generate       Generate synthetic queries via RCP AIaaS"
	@echo "    research-query-generate-local Generate synthetic queries via local LLM (CaaS, GPU required)"
	@echo "    research-query-embed          Embed queries.jsonl.bz2 -> queries-embedded.jsonl.bz2"
	@echo "  Scenario sweep (Run:AI):"
	@echo "    research-list-scenarios            List the registry (id, chunker, chunk_tokens)"
	@echo "    runai-submit-research              SCENARIO=Sx [SWEEP_EXTRA_ARGS=...] one job"
	@echo "    runai-submit-research-all          submit one job per scenario in the registry"
	@echo "    runai-submit-query-embed           [QUERY_EMBED_ARGS=...] one job per study"
	@echo "    runai-submit-query-generate-local  CaaS query-generate via local LLM (one job per study)"
	@echo "  Run:AI debug:"
	@echo "    runai-interactive       Submit a sleep-infinity pod (PVC/GPU/cache check)"
	@echo "    runai-bash              Shell into the debug pod"
	@echo "    runai-delete-debug      Delete the debug pod"
	@echo "  Local utilities:"
	@echo "    s3-fetch                s3://… [TMP_DIR=tmp]  Download + bunzip2 one object"
	@echo
	@echo "  Sweep configuration (override on the make command line):"
	@echo "    STUDY=$(STUDY)  STUDY_CONFIG=$(STUDY_CONFIG)"
	@echo "    SWEEP_EXTRA_ARGS=  forwarded verbatim to impresso-research-embed-sweep"
	@echo
	@echo "  GPU arch selection:"
	@echo "    RUNAI_NODE_POOL  default (a100) | h100 | v100. H200 lives in the h100 pool."
	@echo "    RUNAI_GPU_TYPE   pin a GPU product within the pool (NVIDIA-H100-80GB-HBM3, …)"
	@echo "                     Find labels: kubectl get nodes -L nvidia.com/gpu.product"
	@echo "  Resource requests:"
	@echo "    RUNAI_CPU / RUNAI_CPU_LIMIT / RUNAI_MEMORY / RUNAI_MEMORY_LIMIT"
	@echo
	@echo "  Image tag (defaults to current git branch):"
	@echo "    DOCKER_IMAGE=$(DOCKER_IMAGE_BASE):$(DOCKER_TAG)"
	@echo
	@echo "  Model pin (override in .env.docker):"
	@echo "    CREATOR_NAME=$(CREATOR_NAME) HF_MODEL_NAME=$(HF_MODEL_NAME) HF_MODEL_VERSION=$(HF_MODEL_VERSION)"
	@echo
	@echo "  Detected scenarios: $(SCENARIOS)"
	@echo
	@echo "First-time setup: cp .env.docker.example .env.docker && fill in."
	@echo "S3 + Harbor robot creds go in .env (gitignored)."

.DEFAULT_GOAL := help

.PHONY: help \
        docker-login docker-build docker-build-push docker-push \
        k8s-create-secrets k8s-create-secret k8s-create-pull-secret \
        research-list-scenarios research-corpus-select research-corpus-fetch \
        research-query-generate research-query-generate-local research-query-embed \
        runai-submit-research runai-submit-research-all \
        runai-submit-query-embed runai-submit-query-generate-local \
        runai-interactive runai-bash runai-delete-debug \
        s3-fetch
