"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_scratch_correctness.py`.
"""

import pytest

from vllm.anyscale.anyscale_envs import USE_SCRATCH

MODELS = [
    # "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
]

assert USE_SCRATCH, ("ScratchLLM should be enabled to run a test. "
                     "Use ANYSCALE_VLLM_USE_SCRATCH_LLM=1 pytest -vs "
                     "tests/scratch/anyscale/test_basic_correctness.py")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
        vllm_runner,
        example_prompts,
        model: str,
        dtype: str,
        max_tokens: int,  # not working
) -> None:
    with pytest.raises(AssertionError,
                       match="Scratch only supports page size of 32"):
        vllm_model = vllm_runner(
            model,
            dtype=dtype,
            block_size=16,
        )
    with pytest.raises(AssertionError, match="Only half type is allowed"):
        vllm_model = vllm_runner(
            model,
            dtype="bfloat16",
            block_size=32,
        )
    # The default dtype of llama 3 is bfloat16, so this fails.
    with pytest.raises(AssertionError, match="Only half type is allowed"):
        vllm_model = vllm_runner(
            model,
            dtype="auto",
            block_size=32,
        )
    with pytest.raises(ValueError, match="is not supported by ScratchLLM"):
        vllm_model = vllm_runner(
            model,
            dtype=dtype,
            block_size=32,
            enable_chunked_prefill=True,
        )
    with pytest.raises(ValueError, match="is not supported by ScratchLLM"):
        vllm_model = vllm_runner(
            model,
            dtype=dtype,
            block_size=32,
            enable_prefix_caching=True,
        )
    with pytest.raises(ValueError, match="is not supported by ScratchLLM"):
        vllm_model = vllm_runner(model,
                                 dtype=dtype,
                                 block_size=32,
                                 kv_cache_dtype="fp8")
    print(vllm_model)
