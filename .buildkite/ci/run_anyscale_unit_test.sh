#!/usr/bin/env bash
# This file is to run proprietary unit tests.

set -euo pipefail

# run bash util test
source .buildkite/ci/bash_util/timeout.sh
bash .buildkite/ci/bash_util/test_timeout.sh

# Json tests
# run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/json_constrained_decoding/test_json_constrained_decoding.py
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/anyscale/constrained_decoding
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/anyscale/anyguide
run_with_timeout $(( 15 * 60 )) pytest -vs tests/anyscale/lm_format_enforcer

# Misc
run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/test_sliding_window.py

# Shared memory
run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/shared_memory
