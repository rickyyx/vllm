#!/usr/bin/env bash

# The below benchmarks run on smaller GPUs, e.g. a10g
# Usage: ./run_perf_test.sh

set -euo pipefail

DEFAULT_MAX_NUM_BATCHED_TOKENS=5120

# benchmark 7b
timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-7b \
    --model 7b \
    --tp 1 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 128 --gen-len 512 \
    --request-size 500

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-7b \
    --model 7b \
    --tp 1 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 512 --gen-len 128 \
    --request-size 500

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-7b \
    --model 7b \
    --tp 1 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 2000 --gen-len 2000 \
    --request-size 20

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-7b \
    --model 7b \
    --tp 2 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 128 --gen-len 512 \
    --request-size 500

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-7b \
    --model 7b \
    --tp 2 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 512 --gen-len 128 \
    --request-size 500

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-7b \
    --model 7b \
    --tp 2 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 2000 --gen-len 2000 \
    --request-size 20

# benchmark 13b
timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-13b \
    --model 13b \
    --tp 4 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 128 --gen-len 512 \
    --request-size 500

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-13b \
    --model 13b \
    --tp 4 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 512 --gen-len 128 \
    --request-size 500

timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh \
    --name a10g-13b \
    --model 13b \
    --tp 4 --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
    --prompt-len 2000 --gen-len 2000 \
    --request-size 20
