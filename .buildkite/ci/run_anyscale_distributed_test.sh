#!/usr/bin/env bash

set -euo pipefail

source .buildkite/ci/bash_util/timeout.sh

# Distributed tests.
ANYSCALE_VLLM_TEST_TP=2 DISTRIBUTED_EXECUTOR_BACKEND=ray run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/json_constrained_decoding/test_e2e.py
