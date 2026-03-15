#!/bin/bash
set -e

STRATEGY_DIR="/app/strategy"

# 1. Download strategy artifact if URL is set
if [ -n "$STRATEGY_ARTIFACT_URL" ]; then
    echo "Downloading strategy from $STRATEGY_ARTIFACT_URL"
    ARTIFACT_FILE="/tmp/strategy.zip"

    if [[ "$STRATEGY_ARTIFACT_URL" == s3://* ]]; then
        python -c "
import boto3
url = '$STRATEGY_ARTIFACT_URL'
bucket = url.split('/')[2]
key = '/'.join(url.split('/')[3:])
boto3.client('s3').download_file(bucket, key, '$ARTIFACT_FILE')
print(f'Downloaded from s3://{bucket}/{key}')
"
    elif [[ "$STRATEGY_ARTIFACT_URL" == http* ]]; then
        python -c "
import urllib.request
urllib.request.urlretrieve('$STRATEGY_ARTIFACT_URL', '$ARTIFACT_FILE')
print('Downloaded from $STRATEGY_ARTIFACT_URL')
"
    elif [ -f "$STRATEGY_ARTIFACT_URL" ]; then
        ARTIFACT_FILE="$STRATEGY_ARTIFACT_URL"
    else
        echo "ERROR: Cannot handle artifact URL: $STRATEGY_ARTIFACT_URL"
        exit 1
    fi

    # 2. Deploy using qubx deploy (unzip + uv sync + env setup)
    echo "Deploying strategy..."
    qubx deploy "$ARTIFACT_FILE" -o "$STRATEGY_DIR"
fi

# 3. Build run command
CONFIG="${STRATEGY_CONFIG_PATH:-$STRATEGY_DIR/config.yml}"
ARGS="run $CONFIG"

if [ "$QUBX_PAPER" = "true" ]; then
    ARGS="$ARGS --paper"
fi

if [ -f /app/overrides.yml ]; then
    ARGS="$ARGS --override /app/overrides.yml"
fi

# 4. Run from strategy dir so uv activates the deployed venv
echo "Starting: qubx $ARGS"
cd "$STRATEGY_DIR"
exec uv run qubx $ARGS
