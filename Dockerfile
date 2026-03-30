# ---- Builder: compile qubx wheel from source ----
FROM python:3.12 AS builder

ARG QUBX_VERSION

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY scripts/build.py scripts/build.py
COPY src/ src/

ENV SETUPTOOLS_SCM_PRETEND_VERSION=$QUBX_VERSION
RUN uv build --wheel . --out-dir /wheels

# ---- Runtime: slim image with pip-installed wheel ----
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels/ /tmp/wheels/

# Install qubx wheel with connectors extra (db, tui, k8 are now core deps)
RUN WHEEL=$(ls /tmp/wheels/qubx-*.whl) \
    && pip install --no-cache-dir "${WHEEL}[connectors]" \
    && rm -rf /tmp/wheels

WORKDIR /app

# Pre-initialize ~/.qubx with instruments/fees data so it doesn't run on every container start
RUN python -c "from qubx.core.lookups import FileInstrumentsLookupWithCCXT, FeesLookupFile; FileInstrumentsLookupWithCCXT(); FeesLookupFile()"

COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
