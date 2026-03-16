FROM python:3.12-slim

ARG QUBX_VERSION

# Install qubx with k8 extra (prometheus-client) — no uv needed (wheels installed via pip)
RUN pip install --no-cache-dir "qubx[k8]==${QUBX_VERSION}" boto3

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
