#!/usr/bin/env bash

# Example usage:
# local run:
# bash run.sh --model meta-llama/Llama-2-13b-chat-hf --max_concurrency 20
#
# against staging:
# export ANYSCALE_STAGING_API_BASE="https://console.endpoints-staging.anyscale.com/m/v1"
# export ANYSCALE_STAGING_API_KEY="..."
# bash run.sh --model meta-llama/Llama-2-13b-chat-hf --max_concurrency 20
#

# Initialize variables
MODEL="meta-llama/Llama-2-7b-chat-hf"
MAX_CONCURRENCY="16"

# Do not repeat by default.
REPEAT_DURATION="0m"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift ;;
        --repeat-duration) REPEAT_DURATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if arguments are provided
if [ -z "$MODEL" ] || [ -z "$MAX_CONCURRENCY" ]; then
    echo "Error: Missing arguments. Usage: $0 --model <model_name> --max_concurrency <max_concurrency>"
    exit 1
fi

# Cleanup function
cleanup() {
    echo "run.sh cleanup: received signal $? (fyi that 0==normal exit)"
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID
    fi
    tail -n 50 server.log
    exit 1
}

# Set trap for SIGINT, SIGTERM, and EXIT
trap cleanup SIGINT SIGTERM EXIT

set -e

# Check if ANYSCALE_STAGING_API_BASE is not set
if [ -z "$ANYSCALE_STAGING_API_BASE" ]; then
    echo "ANYSCALE_STAGING_API_BASE env var not set. starting local server"
    python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --num-speculative-tokens 3 \
        --speculative-model JackFram/llama-68m \
        --speculative-model-uses-tp-1 \
        --tensor-parallel-size 4 \
        --enable-cuda-graph \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.80 \
        --num-tokenizer-actors 2 \
        --port 8000 \
        &
    PYTHON_PID=$!

    export ANYSCALE_STAGING_API_BASE="http://localhost:8000/v1"
fi

export OPENAI_API_BASE="$ANYSCALE_STAGING_API_BASE"
export OPENAI_API_KEY="$ANYSCALE_STAGING_API_KEY"

python -u -m vllm.entrypoints.anyscale_ci.stress_test \
    --model "$MODEL" \
    --max-output-len 512 \
    --max-concurrent-requests "$MAX_CONCURRENCY" \
    --repeat-duration $REPEAT_DURATION

set +e

kill $PYTHON_PID
wait

trap - SIGINT SIGTERM EXIT
echo "Script completed successfully."
