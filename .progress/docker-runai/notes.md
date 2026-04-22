# docker-runai — design notes

Containerised execution on EPFL RCP via Run:AI. Reference implementation
lives on the `feat/docker` branch (against the old Make/script layout); this
package adapts it to the new `pyproject.toml`-based code.

## Base image

`nvcr.io/nvidia/pytorch:25.03-py3` (NGC PyTorch). Same choice as
`feat/docker`. Rationale:

- Bundles a CUDA + cuDNN + NCCL + PyTorch stack tuned for NVIDIA hardware
  — the right baseline for an A100 bf16 workload.
- Ships a recent Python (3.12) and `pip` inside `/usr/local/lib/...`, so we
  don't have to bring our own Python.
- We deliberately **do not** install `torch` ourselves: NGC's torch is
  ABI-compatible with the rest of the image (NCCL, transformer-engine,
  apex, …) and re-installing via pip risks breaking that. Our
  `pyproject.toml` declares `torch>=2.2`, which the bundled NGC torch
  satisfies, so `pip install .` is a no-op for torch.

## User / PVC ownership

RCP PVCs are mounted read-write only for the LDAP-matched UID/GID. The
container must `useradd` with that exact UID/GID (otherwise everything we
write is owned by `root` or some other UID and is unreadable by the user
outside the pod).

`Dockerfile` accepts these as build args (`LDAP_UID`, `LDAP_GID`,
`LDAP_USERNAME`, `LDAP_GROUPNAME`) and creates the user. Values come from
`.env.docker` at build time. Find them with `id` on `haas001.rcp.epfl.ch`
(see `.env.docker.example` for the link).

## Entrypoint

`ENTRYPOINT ["impresso-embed-create"]` — the package's own console script.
No bash wrapper. `runai submit ... -- --provider BNL --input-bucket … --output-bucket …`
just appends CLI args. This replaces the old `run.sh` that hard-coded a
provider→aliases dict and called `make` repeatedly: now one provider walks
itself in the CLI.

`HF_HOME` is **not** baked into the image. Set it at submit time
(`--environment HF_HOME=/rcp-scratch/<user>/.cache/hf`) so model weights
land on the persistent PVC instead of inside the ephemeral container FS.

## Why a Makefile (not a Python CLI)

Pure orchestration of external tools (`docker buildx`, `docker push`,
`kubectl`, `runai submit`). No data flowing through. A Makefile is
- short (one shell line per target),
- introspectable (`make help`),
- the same shape as the rest of the EPFL infra docs / `feat/docker`
  conventions.

Wrapping these in Python (Click/Typer) would buy nothing and would cost
indirection. Keep it as Make.

The Makefile is **slim**: only docker + k8s + runai targets. The data-
processing targets from `feat/docker` (sync, newspaper, validate,
provider-stats) are gone — that work is the Python CLI's job now.

## K8s secrets

Two secrets, both created via `kubectl ... --dry-run=client -o yaml | kubectl apply -f -`
so the targets are idempotent (re-runnable when creds rotate):

1. **`s3-credentials`** (generic) — holds `SE_ACCESS_KEY`, `SE_SECRET_KEY`,
   `SE_HOST_URL`. Read by the pod via `--environment KEY=SECRET:s3-credentials,KEY`.
2. **`harbor-pull-secret`** (`docker-registry`) — needed because the image
   lives in a private Harbor project. Created from a Harbor **robot
   account** (username has the form `robot$<project>+<robot-name>`). The
   `$` would be mangled by make's parser, so the recipe shell-sources
   `.env` (`set -a; . ./.env; set +a`) instead of relying on `make`'s
   `-include`.

## Run:AI debug pod

`runai-interactive` submits an `--interactive` job that overrides the
entrypoint with `sleep infinity`. Use it to validate PVC mount, secrets,
GPU presence, model download — anything that a one-shot training job would
fail silently on. Targets `runai-bash` and `runai-delete-debug` are sugar
around `runai bash` / `runai delete`.

`SUPPRESS_DEPRECATION_MESSAGE := true` is exported globally so each `runai`
call doesn't print the v1-CLI deprecation banner.

## Gotchas to remember

- **`docker buildx --platform linux/amd64`** is required when building from
  a Mac — the cluster runs amd64 only.
- **`--load` vs `--push`**: `--load` keeps the image local (good for `docker run` smoke tests on the build host); `--push` ships it straight to the registry. The `docker-build-push` target uses `--push` because pushing a multi-arch buildx image requires it.
- Re-running `k8s-create-secret` updates an existing secret in place
  thanks to `--dry-run=client | kubectl apply`. No need to delete first.
- The `HF_HOME` PVC path must already exist (or be creatable by the LDAP
  user) — Run:AI doesn't `mkdir -p` it. The model loader will, on first
  use, but the **parent** must be writable.

## Out of scope (for now)

- Multi-GPU jobs. `--gpu 1` only.
- Auto-discovering providers from S3 inside the container — caller passes
  `--provider`.
- Building a CPU-only image. Workload is GPU-bound; CPU debugging is done
  locally via `uv run`.
