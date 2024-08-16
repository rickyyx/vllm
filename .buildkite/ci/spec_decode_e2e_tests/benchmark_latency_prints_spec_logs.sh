#!/usr/bin/env bash

set -ex

echo "Running spec benchmark"
# Needs to run enough iterations so that spec logs get printed out (approx. every 5s.)
RAY_DEDUP_LOGS=0 python ./benchmarks/benchmark_latency.py \
    --num-iters 40 \
    --output-len 100 \
    --batch-size 100 \
    --use-sample \
    -tp 1 \
    --model JackFram/llama-68m \
    --speculative-model JackFram/llama-68m \
    --num-speculative-tokens 10 \
    --log-engine-stats \
    2>&1 | tee /mnt/local_storage/benchmark_latency_prints_spec_logs.log

echo "Looking for spec metrics log in output"

if ! grep -q "Speculative metrics:" /mnt/local_storage/benchmark_latency_prints_spec_logs.log; then
    echo "Speculative metrics string not found in the logs."
    exit 1
fi

# Show logs for observability.
grep "Speculative metrics:" /mnt/local_storage/benchmark_latency_prints_spec_logs.log

