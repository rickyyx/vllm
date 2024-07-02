#!/usr/bin/env bash
# This file is to run proprietary unit tests.

set -euo pipefail

# run bash util test
source .buildkite/ci/bash_util/timeout.sh
bash .buildkite/ci/bash_util/test_timeout.sh

# Json tests
# Need to run one by one to avoid GPU OOM.
run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/json_constrained_decoding/test_json_constrained_decoding.py
run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/json_constrained_decoding/test_e2e.py
run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/json_constrained_decoding/test_json_mode_manager.py
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/anyscale/constrained_decoding
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/anyscale/anyguide
run_with_timeout $(( 30 * 60 )) pytest -vs tests/anyscale/lm_format_enforcer

# Misc
run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/test_sliding_window.py

# Shared memory
run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/shared_memory

# ScratchLLM
echo "Run Scratch + vLLM sampling."
ANYSCALE_VLLM_USE_SCRATCH_LLM=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/basic_correctness.py
echo "Run Scratch + Scratch sampling."
ANYSCALE_VLLM_USE_SCRATCH_LLM=1 ANYSCALE_VLLM_USE_SCRATCH_SAMPLE=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/basic_correctness.py
ANYSCALE_VLLM_USE_SCRATCH_LLM=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_input_validation.py
ANYSCALE_VLLM_USE_SCRATCH_LLM=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_prompt_logprob.py
ANYSCALE_VLLM_USE_SCRATCH_LLM=1 run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/scratch/test_compat.py


