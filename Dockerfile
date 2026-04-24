FROM nvcr.io/nvidia/pytorch:25.03-py3

# RCP LDAP user args — required so PVC writes are owned by the right UID/GID.
# Find your EPFL UID/GID with `id` on haas001.rcp.epfl.ch.
ARG LDAP_UID=1000
ARG LDAP_GID=1000
ARG LDAP_USERNAME=user
ARG LDAP_GROUPNAME=usergroup

RUN groupadd -g ${LDAP_GID} ${LDAP_GROUPNAME} \
 && useradd -u ${LDAP_UID} -g ${LDAP_GID} -m -s /bin/bash ${LDAP_USERNAME}

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install the package. NGC's bundled torch satisfies torch>=2.2 — pip will
# leave it alone (we deliberately don't reinstall to keep CUDA/NCCL/apex
# wheel ABI compatibility intact). impresso-essentials is intentionally NOT
# a pyproject dep: its metadata hard-pins numpy==2.2.1 (and dask/pandas…),
# which would uninstall NGC's numpy 1.26.4 and break the ABI stack. The
# three S3 helpers we used from it are vendored into io.py. See
# .progress/io-layer/notes.md.
RUN pip install --no-cache-dir .

# Hard cap transformers at 4.x. transformers>=5 changed model loading to
# meta-device materialization, which frees the storage behind
# `persistent=False` buffers without re-initializing them. Alibaba-NLP's
# custom modeling file (loaded by `trust_remote_code=True` for
# gte-multilingual-base) registers `position_ids` with `persistent=False`,
# so after load it contains uninitialized memory; `rope_cos[position_ids]`
# then trips a CUDA IndexKernel assert on any input, including a 6-token
# smoke test. Upstream: HF transformers #43950 / #44534; model discussion
# #30. Alibaba-NLP/new-impl is unmaintained (last commit Aug 2024) so the
# fix has to live on our side. Also cap sentence-transformers below 5.2:
# 5.2 is the first line that pulls transformers v5 by default on fresh
# resolves. pyproject.toml carries the same caps — this belt-and-suspenders
# RUN protects against pip resolver drift and makes the reason visible at
# the image layer. See .progress/transformers-v5-regression/notes.md.
RUN pip install --no-cache-dir \
      "transformers>=4.46,<5" \
      "sentence-transformers>=5.0,<5.2"

# Build-time guardrail: fail the build if anything silently upgraded numpy
# past the NGC-bundled 1.26.x (which would break apex/NCCL/TE ABI).
RUN python -c "import numpy; assert numpy.__version__.startswith('1.26'), f'numpy was upgraded to {numpy.__version__}'"

# Build-time guardrail: fail the build if the transformers cap slipped.
RUN python -c "import transformers; v=transformers.__version__; assert v.startswith('4.'), f'transformers was upgraded to {v} — see .progress/transformers-v5-regression/notes.md'"

ENV PYTHONUNBUFFERED=1

USER ${LDAP_USERNAME}

# Override HF_HOME at runtime to point at the PVC, e.g.
#   --environment HF_HOME=/rcp-scratch/<user>/.cache/hf
# Args are appended verbatim by `runai submit ... -- <args>`.
ENTRYPOINT ["impresso-embed-create"]
