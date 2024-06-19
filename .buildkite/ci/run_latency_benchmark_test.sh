#!/usr/bin/env bash
# Run benchmarks we use in development to ensure they work.

set -euo pipefail

# Validate profiling (+ ray workers) doesn't crash.
RAY_DEDUP_LOGS=0 python ./benchmarks/benchmark_latency.py \
    --num-iters 2 \
    --output-len 20 \
    --batch-size 1 \
    --model facebook/opt-125m \
    -tp 2 \
    --enable-cuda-graph \
    --enable-custom-nccl \
    --profile \
    --profile-ray-workers \
    --profile-logdir /mnt/local_storage/ \
    --disable-shared-memory

# Validate profiling (no ray workers) doesn't crash.
python ./benchmarks/benchmark_latency.py \
    --num-iters 2 \
    --output-len 20 \
    --batch-size 1 \
    --model facebook/opt-125m  \
    -tp 1 \
    --enable-cuda-graph \
    --profile \
    --profile-logdir /mnt/local_storage/

# Validate profiling with speculative decoding doesn't crash.
RAY_DEDUP_LOGS=0 python ./benchmarks/benchmark_latency.py \
    --num-iters 2 \
    --output-len 20 \
    --batch-size 1 \
    --model facebook/opt-125m \
    --speculative-model facebook/opt-125m \
    --num-speculative-tokens 5 \
    -tp 2 \
    --enable-cuda-graph \
    --enable-custom-nccl \
    --profile \
    --profile-ray-workers \
    --profile-logdir /mnt/local_storage/ \
    --disable-shared-memory
