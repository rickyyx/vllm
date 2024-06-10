import os
import random
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import BaseModel

from tests.anyscale.json_constrained_decoding.utils import (
    FaultyJsonModeLogitsProcessor, FaultyProcessorWithException)
from tests.worker.utils import (create_execute_model_data,
                                create_seq_group_metadata_from_prompts,
                                create_worker)
from vllm.anyscale.constrained_decoding.json_mode_manager import (
    JSONModeManager)
from vllm.config import LogitProcessorConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.model_executor.input_metadata import InputMetadata
from vllm.sequence import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.worker.worker import Worker

MODEL_ID = "JackFram/llama-68m"
VOCAB_SIZE = 32000

NUM_FAULT_TOLERANT_ITERATIONS = 1000


class Gender(BaseModel):
    value: Literal["male", "female", "other"]


class AnswerFormat(BaseModel):
    age: int
    name: str
    gender: Gender
    profession: str
    hobbies: list[str]


EXAMPLE_COMPLETION = AnswerFormat(
    age=25,
    name="John",
    gender=Gender(value="male"),
    profession="engineer",
    hobbies=["reading", "swimming", "hiking"],
).json()


def get_input_metadata(model_id: str, bsize: int) -> InputMetadata:
    """Prepare InputMetadata that can be passed to the logit processor.

    This function emulates being in the middle of generation for bsize
    concurrent requests all of which have the same schema, but are at different
    state of generation.

    We generate bsize prompts at random lengths with random numbers, and then
    given a fixed completion, we emulate variable length partial generations of
    the output.

    Args:
        model_id: The model identifier.
        bsize: The number of concurrent requests.
    Returns:
        InputMetadata: The input metadata that can be passed to the logit
        processor.
    """
    seed = 100
    tokenizer = get_tokenizer(model_id)
    block_size = 32
    num_gpu_blocks = 512 // block_size * bsize
    random.seed(seed)

    worker = create_worker(
        Worker,
        model_name=model_id,
        seed=seed,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
    )

    # Generate bsize prompts at different lengths between 32, 512
    prompt_lens = [random.randint(32, 512) for _ in range(bsize)]
    prompt_tokens = [random.randint(0, 32000) for _ in range(1000)]
    completion_tokens = tokenizer.encode(EXAMPLE_COMPLETION,
                                         add_special_tokens=False)

    # Sample subsequences of the prompt tokens
    prompts = []
    for prompt_len in prompt_lens:
        start_idx = random.randint(0, len(prompt_tokens) - prompt_len)
        prompts.append(prompt_tokens[start_idx:start_idx + prompt_len])

    num_prev_gen_tokens = [
        random.randint(1, len(completion_tokens)) for _ in range(bsize)
    ]
    prev_output_tokens = [
        completion_tokens[:gen_len] for gen_len in num_prev_gen_tokens
    ]
    final_seq_lens = [
        len(prompt + output_tokens) + 1
        for prompt, output_tokens in zip(prompts, prev_output_tokens)
    ]

    sampling_param = SamplingParams(
        temperature=0.0,
        response_format={
            "type": "json",
            "schema": AnswerFormat.schema_json()
        },
    )

    print(f"{prompts=}")
    print(f"{prev_output_tokens=}")

    execute_model_data = create_execute_model_data(
        create_seq_group_metadata_from_prompts(
            prompts,
            num_gpu_blocks,
            block_size,
            continuations=prev_output_tokens,
            final_seq_lens=final_seq_lens,
            sampling_params=sampling_param,
        ))

    seq_group_metadata_list = execute_model_data.seq_group_metadata_list
    _, _, input_metadata, *_ = worker._prepare_inputs(seq_group_metadata_list)  # pylint: disable=protected-access

    return input_metadata


@pytest.mark.parametrize(
    "batch_and_actor_size",
    [
        pytest.param((1, 1), marks=pytest.mark.timeout(60)),
        pytest.param((1, 8), marks=pytest.mark.timeout(60)),
        pytest.param((32, 8), marks=pytest.mark.timeout(120)),
        pytest.param((32, 16), marks=pytest.mark.timeout(120)),
    ],
)
def test_logit_processor_manager_fault_tolerance(batch_and_actor_size):
    """Test logit processor manager fault tolerance.

    LogitProcessor raises an exception on every call with prob p.
    We should be able to recover from these exceptions and continue without
    interruption. This test should timeout in a few minutes if the fault
    tolerance is not working or is taking longer than expected.

    You can monitor Ray Dashboard for restarted actors, you should see a lot of
    restarted actors when this test runs.
    """
    bsize, actor_count = batch_and_actor_size

    with patch.dict(
            os.environ,
        {"ANYSCALE_VLLM_NUM_JSON_LOGITS_PROCESSOR_ACTORS": str(actor_count)}):
        ft_config = LogitProcessorConfig(
            logit_processor_cls=FaultyProcessorWithException,
            recreate_failed_actors=True,
            max_restarts=-1,
            delay_between_actor_restarts_s=0.0,
        )

        manager = JSONModeManager(
            tokenizer_name_or_path=MODEL_ID,
            vocab_size=VOCAB_SIZE,
            logit_processor_config=ft_config,
        )

        logits = torch.rand(bsize, VOCAB_SIZE)
        input_metadata = get_input_metadata(MODEL_ID, bsize)

        iteration_cnt = 0
        at_least_one_failure = False
        while iteration_cnt < NUM_FAULT_TOLERANT_ITERATIONS:
            print(f"Iteration {iteration_cnt} ...")
            buffer_inds_to_batch_inds = manager.start_json_logits_processors(
                logits, input_metadata)
            _, valid_mask = manager.get_json_logits_bias(
                logits,
                buffer_inds_to_batch_inds,
            )

            if valid_mask is not None and not torch.all(valid_mask):
                at_least_one_failure = True
            iteration_cnt += 1

        assert at_least_one_failure, (
            "At least one failure should have occurred.")


def test_engine_e2e_fault_tolerance():
    """Tests end-to-end fault tolerance of the engine.

    Send two requests to an engine with a faulty logit processor. Run through
    the decoding, at some point the requests will fail and the engine should
    recover and continue without interruption. There should be at least one
    request that receives an exception object during the process.
    """

    engine_args = EngineArgs(
        model=MODEL_ID,
        enable_json_logits_processors=True,
        # This faulty logit processor has p_error = p_timeout = 0.01
        logit_processor_cls=FaultyJsonModeLogitsProcessor,
        json_fault_tolerance_recreate_failed_actors=True,
        json_fault_tolerance_delay_beteen_actor_restarts_s=0.0,
        json_fault_tolerance_max_restarts=-1,
    )

    engine = LLMEngine.from_engine_args(engine_args)
    prompts = [
        "What is your name? My name is John. I am an engineer. I like reading,"
        "swimming and hiking. I am 25 years old.",
        "What is your age? My name is Sarah, I am 30 years old. I am a doctor."
        "I like playing tennis and reading.",
    ]
    engine.add_request(
        request_id="0",
        prompt=prompts[0],
        sampling_params=SamplingParams(
            temperature=0.0,
            response_format={
                "type": "json",
                "schema": AnswerFormat.schema_json()
            },
            stop=["</s>"],
            max_tokens=256,
        ),
    )

    engine.add_request(
        request_id="1",
        prompt=prompts[1],
        sampling_params=SamplingParams(
            temperature=0.0,
            response_format={
                "type": "json",
                "schema": AnswerFormat.schema_json()
            },
            stop=["</s>"],
            max_tokens=256,
        ),
    )

    at_least_one_failure = False
    while engine.has_unfinished_requests():
        step_output = engine.step()
        for output in step_output:
            if isinstance(output, Exception):
                at_least_one_failure = True
                continue
    assert at_least_one_failure, "At least one failure should have occurred."


@patch("importlib.import_module")
def test_logit_processor_config_with_str_input(mock_import_module):
    """Tests that the config can be init with a str input."""

    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.mock_class = mock_class

    mock_import_module.return_value = mock_module

    config = LogitProcessorConfig(logit_processor_cls="mock_module.mock_class")
    mock_import_module.assert_called_with("mock_module")
    assert config.logit_processor_cls == mock_class, (
        "logit_processor_cls should be a class type when initialized "
        "with a string.")
