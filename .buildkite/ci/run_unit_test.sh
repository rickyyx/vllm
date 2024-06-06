#!/usr/bin/env bash

set -euo pipefail

# run bash util test
source .buildkite/ci/bash_util/timeout.sh
bash .buildkite/ci/bash_util/test_timeout.sh

# run unit test
# run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/json_constrained_decoding/test_json_constrained_decoding.py
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/anyscale/constrained_decoding
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/anyscale/anyguide
# run_with_timeout $(( 15 * 60 )) pytest -vs tests/anyscale/lm_format_enforcer
# Regression test
run_with_timeout $(( 60 )) pytest -v -s tests/test_regression.py
# Basic correctness test
run_with_timeout $(( 60 * 10 )) VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s tests/basic_correctness/test_basic_correctness.py
run_with_timeout $(( 60 * 10 )) VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest -v -s tests/basic_correctness/test_basic_correctness.py
run_with_timeout $(( 60 * 10 )) VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s tests/basic_correctness/test_chunked_prefill.py
run_with_timeout $(( 60 * 10 )) VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest -v -s tests/basic_correctness/test_chunked_prefill.py
run_with_timeout $(( 60 * 10 )) VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 pytest -v -s tests/basic_correctness/test_preemption.py
# Entrypoints Test
run_with_timeout $(( 60 * 20 )) pytest -v -s tests/entrypoints -m llm
run_with_timeout $(( 60 * 20 )) pytest -v -s tests/entrypoints -m openai
# Lora Test
run_with_timeout $(( 60 * 240 )) pytest -v -s tests/lora --ignore=lora/test_long_context.py
# Spec decode test
run_with_timeout $(( 60 * 60 )) pytest -v -s tests/spec_decode

# run_with_timeout $(( 60 )) pytest -vs tests/async_engine
# run_with_timeout $(( 60 )) pytest -vs tests/core
# run_with_timeout $(( 15 * 60 )) pytest -vs tests/engine
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/worker
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/samplers
# run_with_timeout $(( 40 * 60 )) pytest -vs tests/anyscale/lora --ignore=tests/anyscale/lora/test_long_context.py --ignore=tests/anyscale/lora/test_layer_variation.py
# run_with_timeout $(( 40 * 60 )) pytest -vs tests/anyscale/test_async_decoding.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/test_utils.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/test_sliding_window.py
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/test_sampler.py
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/anyscale/ops
# run_with_timeout $(( 20 * 60 )) pytest -vs tests/anyscale/test_distributed.py
# run_with_timeout $(( 10 * 60 )) pytest -vs tests/anyscale/shared_memory
# run_with_timeout $(( 60 )) pytest -vs tests/test_sequence.py
# run_with_timeout $(( 5 * 60 )) pytest -vs tests/multimodal/test_llava_modeling.py
# run_with_timeout $(( 40 * 60 )) pytest -vs tests/test_multi_decoding.py

# run benchmarks
run_with_timeout $(( 20 * 60 )) bash ./benchmarks/anyscale/benchmark.sh
