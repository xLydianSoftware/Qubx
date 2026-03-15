FROM python:3.12-slim

ARG QUBX_VERSION

# Install qubx with k8 extra (prometheus-client) and uv for strategy dep management
RUN pip install --no-cache-dir "qubx[k8]==${QUBX_VERSION}" uv boto3

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
