#!/usr/bin/env bash

set -euo pipefail
source .buildkite/ci/bash_util/timeout.sh

run_with_timeout $(( 60 * 60 )) pytest -vvs tests/spec_decode/test_smoke.py
run_with_timeout $(( 60 * 60 )) pytest -vvs tests/spec_decode/test_integration.py

run_with_timeout $(( 5 * 60 )) bash .buildkite/ci/spec_decode_e2e_tests/benchmark_latency_prints_spec_logs.sh
