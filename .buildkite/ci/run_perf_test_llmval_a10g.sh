#!/usr/bin/env bash

# The below benchmarks run on smaller GPUs, e.g. a10g
# Usage: ./run_perf_test_llmval_a10g.sh

set -euo pipefail

DEFAULT_MAX_NUM_BATCHED_TOKENS=5120

tps=("1" "2" "4")
concurrencies=("5")

# benchmark 7b
echo "--- :python: Running benchmarks for 7b"
for tp in "${tps[@]}"; do
  if [ "$tp" -gt 1 ]; then
    ENABLE_CUSTOM_NCCL="--enable-custom-nccl"
  else
    ENABLE_CUSTOM_NCCL=""
  fi

  for concurrency in "${concurrencies[@]}"; do
    timeout $(( 60 * 60 )) bash ./benchmarks/anyscale/benchmark_llmval.sh \
        --name a10-7b \
        --model 7b \
        --tp $tp --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
        --min-lines 15  --max-lines 50 --num-users $concurrency \
        --enable-cuda-graph $ENABLE_CUSTOM_NCCL \
        --request-size 150
  done
done

# No tp=1 for 13b since it doens't fit on a10g.
tps=("2" "4")
echo "--- :python: Running benchmarks for 13b"
for tp in "${tps[@]}"; do
  for concurrency in "${concurrencies[@]}"; do
    timeout $(( 60 * 60 )) bash ./benchmarks/anyscale/benchmark_llmval.sh \
        --name a10-13b \
        --model 13b \
        --tp $tp --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
        --min-lines 15  --max-lines 50 --num-users $concurrency \
        --enable-cuda-graph $ENABLE_CUSTOM_NCCL \
        --request-size 150
  done
done