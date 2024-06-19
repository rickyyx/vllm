#!/usr/bin/env bash

set -euo pipefail

source .buildkite/ci/bash_util/timeout.sh

# run correctness test
run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/cuda_graph
run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/chunked_prefill
run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/test_sliding_window.py
