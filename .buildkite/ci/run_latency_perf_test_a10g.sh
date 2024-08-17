#!/usr/bin/env bash

# The below benchmarks run on smaller GPUs, e.g. a10g

set -euo pipefail

PROFILE_LOGDIR=/tmp/profile_logs

# benchmark 7b latency with profiling
MODEL="7b"
MODEL_DOWNLOAD_DIR="/mnt/local_storage"
source ./benchmarks/download_model.sh
LOCAL_PATH="${MODEL_DOWNLOAD_DIR}/llama-${MODEL}/"
download_weights "$MODEL" "$LOCAL_PATH"

# Set CUDA_LAUNCH_BLOCKING for synchronous Cuda operations.
export CUDA_LAUNCH_BLOCKING=1

# Set output-len to 32 so that the trace is not too large and can be loaded.
timeout $(( 20 * 60 )) python ./benchmarks/benchmark_latency.py \
    --model=$LOCAL_PATH \
    --use-dummy-weights \
    --tensor-parallel-size=1 \
    --output-len=32 \
    --num-iters=1 \
    --profile \
    --profile-logdir=$PROFILE_LOGDIR

# Upload the log artifacts to S3.
aws s3 cp "$PROFILE_LOGDIR" "$ANYSCALE_ARTIFACT_STORAGE/vllm-ci-artifacts/" --recursive
