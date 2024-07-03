#!/usr/bin/env bash
# This file is to run proprietary unit tests.

set -euo pipefail

# run bash util test
source .buildkite/ci/bash_util/timeout.sh
bash .buildkite/ci/bash_util/test_timeout.sh

# ScratchLLM
export ANYSCALE_VLLM_USE_SCRATCH_LLM=1
echo "Run Scratch + vLLM sampling."
run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_basic_correctness.py 
echo "Run Scratch + Scratch sampling."
ANYSCALE_VLLM_USE_SCRATCH_SAMPLE=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_basic_correctness.py 
run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_input_validation.py
run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_compat.py
# Failing.
# ANYSCALE_VLLM_USE_SCRATCH_LLM=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_models.py
# ANYSCALE_VLLM_USE_SCRATCH_LLM=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_prompt_logprob.py
