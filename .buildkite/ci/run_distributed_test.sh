#!/usr/bin/env bash

set -euo pipefail

# Download images for Llava distributed tests
pip install awscli  # workaround for https://github.com/aws/aws-cli/issues/8435
aws s3 sync s3://endpoints-llava-test/images/ tests/images

source .buildkite/ci/bash_util/timeout.sh

# Distributed tests.
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
# TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
# TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
# TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=facebook/opt-125m DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
# TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf DISTRIBUTED_EXECUTOR_BACKEND=mp run_with_timeout $(( 20 * 60 )) pytest -v -s tests/distributed/test_chunked_prefill_distributed.py
pytest -v -s tests/spec_decode/e2e/test_integration_dist.py 
