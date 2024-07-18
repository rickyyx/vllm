from typing import List

import pytest
import ray
import torch

from vllm import SamplingParams

from vllm.anyscale.anyscale_envs import USE_SCRATCH

assert USE_SCRATCH, ("ScratchLLM should be enabled to run a test. "
                     "Use ANYSCALE_VLLM_USE_SCRATCH_LLM=1 pytest -vs "
                     "tests/scratch/anyscale/test_basic_correctness.py")

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Meta-Llama-3-8B",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("num_top_logprobs", [6])  # 32000 == vocab_size
@pytest.mark.parametrize("detokenize", [True, False])
def test_get_prompt_logprobs(
    hf_runner,
    vllm_runner,
    model,
    dtype,
    num_top_logprobs: int,
    detokenize: bool,
    example_prompts,
):
    atol = 1e-2
    # TODO(sang): Currently error rate is pretty high.
    rtol = 0.2
    max_tokens = 5

    # Temporary workaround to fix Scratch issue.
    @ray.remote(num_gpus=1)
    def f():
        with hf_runner(model, dtype=dtype) as hf_model:
            hf_logprobs = hf_model.generate_greedy_logprobs(
                example_prompts,
                max_tokens=max_tokens,
            )
            return hf_logprobs

    hf_logprobs = ray.get(f.remote())

    with vllm_runner(
            model,
            dtype=dtype,
            max_logprobs=num_top_logprobs,
            block_size=32,
    ) as vllm_model:
        vllm_sampling_params = SamplingParams(max_tokens=max_tokens,
                                              logprobs=num_top_logprobs,
                                              prompt_logprobs=num_top_logprobs,
                                              temperature=0.0,
                                              detokenize=detokenize)
        vllm_results = vllm_model.model.generate(
            example_prompts, sampling_params=vllm_sampling_params)

    # Test whether logprobs are included in the results.
    for result in vllm_results:
        assert result.prompt_logprobs is not None
        assert result.outputs[0].logprobs is not None
        assert len(result.outputs[0].logprobs) == max_tokens
        for logprobs in result.outputs[0].logprobs:
            assert len(logprobs) == num_top_logprobs
        output_text = result.outputs[0].text
        output_string_from_most_likely_tokens_lst: List[str] = []
        for top_logprobs in result.outputs[0].logprobs:
            top_logprob = next(iter(top_logprobs.values()))
            output_string_from_most_likely_tokens_lst.append(
                top_logprob.decoded_token)

        if detokenize:
            output_string_from_most_likely_tokens = "".join(
                output_string_from_most_likely_tokens_lst)
            assert output_text == output_string_from_most_likely_tokens, (
                "The output text from the top logprob for each token position "
                "should be the same as the output text in the result.")
        else:
            assert output_text == ''
            assert output_string_from_most_likely_tokens_lst == ([None] *
                                                                 max_tokens)

        # The first prompt logprob is always None
        assert result.prompt_logprobs[0] is None
        for prompt_logprobs in result.prompt_logprobs[1:]:
            # If the prompt token is not included in the top X
            # logprob, it can return 1 more data
            assert (len(prompt_logprobs) == num_top_logprobs
                    or len(prompt_logprobs) == num_top_logprobs + 1)

    # Test whether prompt logprobs are consistent with HF
    for vllm_result, hf_logprob in zip(vllm_results, hf_logprobs):
        # Check prompt logprobs
        # The first prompt logprob is always None, so we compare it from 1:.
        vllm_prompt_logprobs = vllm_result.prompt_logprobs[1:]
        for i, vllm_prompt_logprob_dict in enumerate(vllm_prompt_logprobs):
            for token_id, logprob in vllm_prompt_logprob_dict.items():
                torch.testing.assert_close(logprob.logprob,
                                           hf_logprob[0][i][token_id].item(),
                                           atol=atol,
                                           rtol=rtol)
        vllm_sample_logprobs = vllm_result.outputs[0].logprobs
        for i, top_logprobs in enumerate(vllm_sample_logprobs):
            for token_id, sample_logprob in top_logprobs.items():
                logprob = sample_logprob.logprob
                torch.testing.assert_close(logprob,
                                           hf_logprob[i][-1][token_id].item(),
                                           atol=atol,
                                           rtol=rtol)
                if detokenize:
                    assert isinstance(sample_logprob.decoded_token, str), (
                        "The token should be decoded by the time it is returned"
                        " to the user.")

    # Test if prompt logprobs are correctly set.
    for vllm_result in vllm_results:
        token_ids = vllm_result.prompt_token_ids
        prompt_logprobs = vllm_result.prompt_logprobs

        # The first token doesn't have logprob.
        assert prompt_logprobs[0] is None

        for token_id, logprob_dict in zip(token_ids[1:], prompt_logprobs[1:]):
            assert token_id in logprob_dict
