#!/usr/bin/env bash

# The below benchmarks run on larger GPUs, e.g. a100
# Usage: ./run_perf_test_llmval_a100.sh

set -uo pipefail

DEFAULT_MAX_NUM_BATCHED_TOKENS=5120

tps=("1" "2" "4" "8")
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
        --name a100-7b \
        --model 7b \
        --tp $tp --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
        --min-lines 15  --max-lines 50 --num-users $concurrency \
        --enable-cuda-graph $ENABLE_CUSTOM_NCCL \
        --request-size 150
  done
done


# benchmark 13B
echo "--- :python: Running benchmarks for 13b"
for tp in "${tps[@]}"; do
  if [ "$tp" -gt 1 ]; then
    ENABLE_CUSTOM_NCCL="--enable-custom-nccl"
  else
    ENABLE_CUSTOM_NCCL=""
  fi

  for concurrency in "${concurrencies[@]}"; do
    timeout $(( 60 * 60 )) bash ./benchmarks/anyscale/benchmark_llmval.sh \
        --name a100-13b \
        --model 13b \
        --tp $tp --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
        --min-lines 15  --max-lines 50 --num-users $concurrency \
        --enable-cuda-graph $ENABLE_CUSTOM_NCCL \
        --request-size 150

  done
done


# benchmark 70b
# Don't run tp=1 for 70b since it doesn't fit on a100.
tps=("2" "4" "8")
echo "--- :python: Running benchmarks for 70b"
for tp in "${tps[@]}"; do
  for concurrency in "${concurrencies[@]}"; do
    timeout $(( 60 * 60 )) bash ./benchmarks/anyscale/benchmark_llmval.sh \
        --name a100-70b \
        --model 70b \
        --tp $tp --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
        --min-lines 15  --max-lines 50 --num-users $concurrency \
        --enable-cuda-graph $ENABLE_CUSTOM_NCCL \
        --request-size 150

    timeout $(( 60 * 60 )) bash ./benchmarks/anyscale/benchmark_llmval.sh \
        --name a100-70b \
        --model 70b \
        --tp $tp --max-num-batched-tokens $DEFAULT_MAX_NUM_BATCHED_TOKENS \
        --min-lines 250  --max-lines 251 --num-users $concurrency \
        --enable-cuda-graph $ENABLE_CUSTOM_NCCL \
        --request-size 150
  done
done