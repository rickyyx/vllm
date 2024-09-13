"""Verify Scratch won't crash with large batches for long running.
"""

import pytest

from vllm.anyscale.anyscale_envs import USE_SCRATCH

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Meta-Llama-3-8B",
    # "meta-llama/Meta-Llama-3-8B-Instruct",
]

assert USE_SCRATCH, ("ScratchLLM should be enabled to run a test. "
                     "Use ANYSCALE_VLLM_USE_SCRATCH_LLM=1 pytest -vs "
                     "tests/scratch/anyscale/test_basic_correctness.py")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [64])
def test_session_gc(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    example_prompts = example_prompts
    batch_size = 8
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        block_size=32,
        max_num_seqs=batch_size,
        disable_async_output_proc=True,
    )

    worker = vllm_model.model.llm_engine.model_executor.driver_worker
    runner = worker.model_runner
    cache = runner._scratch_session_manager._session_cache
    for _ in range(5):
        vllm_model.generate_greedy(example_prompts, max_tokens)
        # Verify all sessions are GC'ed.
        # This test relies on some implementation detail which is not good.
        # But idk what's the best way to test it otherwise.
        # Since GC of sessions happen before the next model execution,
        # and we batch batch_size requests at a time, we should have only 8
        # sessions left.
        assert len(cache) <= batch_size
