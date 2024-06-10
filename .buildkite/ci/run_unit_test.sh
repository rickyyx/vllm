#!/usr/bin/env bash

set -euo pipefail

# run bash util test
source .buildkite/ci/bash_util/timeout.sh
bash .buildkite/ci/bash_util/test_timeout.sh

# Regression test
run_with_timeout $(( 60 )) pytest -v -s tests/test_regression.py

# Basic correctness test
run_with_timeout $(( 60 * 10 )) pytest -v -s tests/basic_correctness/test_basic_correctness.py
run_with_timeout $(( 60 * 10 )) pytest -v -s tests/basic_correctness/test_chunked_prefill.py
VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 run_with_timeout $(( 60 * 10 )) pytest -v -s tests/basic_correctness/test_preemption.py

# Entrypoints Test
run_with_timeout $(( 60 * 20 )) pytest -v -s tests/entrypoints -m llm
run_with_timeout $(( 60 * 20 )) pytest -v -s tests/entrypoints -m openai
# Lora Test
# run_with_timeout $(( 60 * 240 )) pytest -v -s tests/lora --ignore=lora/test_long_context.py
# Spec decode test
run_with_timeout $(( 60 * 60 )) pytest -v -s tests/spec_decode

# run benchmarks
# run_with_timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh
