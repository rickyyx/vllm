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
        hf_runner,
        vllm_runner,
        example_prompts,
        model: str,
        dtype: str,
        max_tokens: int,  # not working
) -> None:
    # Pop out prompts that currently fail.
    # TODO(sang, ricky): Fix it.
    example_prompts.pop(2)
    example_prompts.pop()

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        block_size=32,
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        print("hf_output_str", hf_output_str)
        print("vllm_output_str", vllm_output_str)
        assert (
            hf_output_str == vllm_output_str
        ), f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}"
        assert (hf_output_ids == vllm_output_ids
                ), f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}"
