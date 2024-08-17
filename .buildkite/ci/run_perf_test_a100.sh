#!/usr/bin/env bash

# The below benchmarks run on larger GPUs, e.g. a100
# Usage: ./run_perf_test.sh

set -euo pipefail

DEFAULT_MAX_NUM_BATCHED_TOKENS=5120

# benchmark 13b
timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-13b \
    --model 13b \
    --tp 2 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 128 --gen-len 512 --max-num-batched-tokens 20000 \
    --request-size 500  --enable-cuda-graph --enable-custom-nccl

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-13b \
    --model 13b \
    --tp 2 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 512 --gen-len 128 --max-num-batched-tokens 20000 \
    --request-size 500  --enable-cuda-graph --enable-custom-nccl

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-13b \
    --model 13b \
    --tp 2 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 2000 --gen-len 2000 --max-num-batched-tokens 20000 \
    --request-size 20  --enable-cuda-graph --enable-custom-nccl


# benchmark 70b
# using larger max-num-batched-tokens to increase prefilling speed
timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-70b \
    --model 70b \
    --tp 4 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 128 --gen-len 512 \
    --request-size 500 --enable-cuda-graph --enable-custom-nccl

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-70b \
    --model 70b \
    --tp 4 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 512 --gen-len 128 \
    --request-size 500 --enable-cuda-graph --enable-custom-nccl

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-70b \
    --model 70b \
    --tp 4 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 2000 --gen-len 2000 \
    --request-size 20 --enable-cuda-graph --enable-custom-nccl


timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-70b \
    --model 70b \
    --tp 8 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 3500 --gen-len 150 \
    --request-size 20 --enable-cuda-graph --enable-custom-nccl

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a100-70b \
    --model 70b \
    --tp 8 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 3500 --gen-len 500 \
    --request-size 20 --enable-cuda-graph --enable-custom-nccl
