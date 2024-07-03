#!/usr/bin/env bash

set -euo pipefail

# run bash util test
source .buildkite/ci/bash_util/timeout.sh
bash .buildkite/ci/bash_util/test_timeout.sh

# Regression test
run_with_timeout $(( 120 )) pytest -v -s tests/test_regression.py

# Basic correctness test
run_with_timeout $(( 60 * 10 )) pytest -v -s tests/basic_correctness/test_basic_correctness.py
run_with_timeout $(( 60 * 10 )) pytest -v -s tests/basic_correctness/test_chunked_prefill.py
VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 run_with_timeout $(( 60 * 10 )) pytest -v -s tests/basic_correctness/test_preemption.py

# Entrypoints Test
# RAY_OVERRIDE_JOB_RUNTIME_ENV=1 is required because these tests specify runtime env itself.
RAY_OVERRIDE_JOB_RUNTIME_ENV=1 run_with_timeout $(( 60 * 20 )) pytest -v -s tests/entrypoints/openai
RAY_OVERRIDE_JOB_RUNTIME_ENV=1 run_with_timeout $(( 60 * 20 )) pytest -v -s tests/entrypoints/llm
# Lora Test
# run_with_timeout $(( 60 * 240 )) pytest -v -s tests/lora --ignore=lora/test_long_context.py
# Spec decode test
VLLM_ATTENTION_BACKEND=XFORMERS run_with_timeout $(( 60 * 60 )) pytest -v -s tests/spec_decode

# run benchmarks
# run_with_timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh
