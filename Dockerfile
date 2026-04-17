FROM nvcr.io/nvidia/pytorch:26.03-py3

# RCP LDAP user args — required to avoid ownership issues on PVC mounts
# Obtain your EPFL UID/GID: https://wiki.rcp.epfl.ch/en/home/CaaS/FAQ/how-to-find-uid-gid
ARG LDAP_UID=1000
ARG LDAP_GID=1000
ARG LDAP_USERNAME=user
ARG LDAP_GROUPNAME=usergroup

# System deps: make is required by the Makefile
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create LDAP-matched group and user
RUN groupadd -g ${LDAP_GID} ${LDAP_GROUPNAME} && \
    useradd -u ${LDAP_UID} -g ${LDAP_GID} -m -s /bin/bash ${LDAP_USERNAME}

# Copy project code into user home (RCP convention)
WORKDIR /home/${LDAP_USERNAME}
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Alibaba-NLP/gte-multilingual-base at the pinned revision
# This avoids network calls at job start on the cluster
ENV HF_HOME=/opt/hf_cache
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', \
    revision='f7d567e', trust_remote_code=True); \
print('Model cached, max_seq_length:', m.max_seq_length)"

# Give the LDAP user ownership of code and model cache
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME} /opt/hf_cache

ENV PYTHONUNBUFFERED=1
USER ${LDAP_USERNAME}

# Usage: docker run <image> <PROVIDER> <GPU_NUMBER>
# Example: docker run <image> BNL 0
ENTRYPOINT ["bash", "run.sh"]
