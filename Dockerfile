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
# wheel ABI compatibility intact).
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

USER ${LDAP_USERNAME}

# Override HF_HOME at runtime to point at the PVC, e.g.
#   --environment HF_HOME=/rcp-scratch/<user>/.cache/hf
# Args are appended verbatim by `runai submit ... -- <args>`.
ENTRYPOINT ["impresso-embed-create"]
