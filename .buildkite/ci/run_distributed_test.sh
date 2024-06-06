#!/usr/bin/env bash

set -euo pipefail

# Download images for Llava distributed tests
pip install awscli  # workaround for https://github.com/aws/aws-cli/issues/8435
aws s3 sync s3://endpoints-llava-test/images/ tests/images

source .buildkite/ci/bash_util/timeout.sh

# Distributed tests.
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
pytest -v -s tests/spec_decode/e2e/test_integration_dist.py 


# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/chunked_prefill
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/test_distributed.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/cuda_graph/test_correctness.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/cuda_graph/test_api.py
# run_with_timeout $(( 3 * 60 * 60 )) pytest -vs tests/anyscale/lora/test_layer_variation.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/lora/test_long_context.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/spec_decode/test_integration.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/multimodal/test_llava_integration.py

run_with_timeout $(( 5 * 60 )) bash .buildkite/ci/run_latency_benchmark_test.sh

# Run only one iteration of stress test.
run_with_timeout $(( 10 * 60 )) bash .buildkite/ci/sampling_params_stress_test/run.sh \
    --model JackFram/llama-160m \
    --max-concurrency 32

# Run a small version of ShareGPT benchmark.
run_with_timeout $((5 * 60)) bash benchmarks/anyscale/sharegpt/run_in_ci.sh
