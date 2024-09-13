import asyncio
import os
import random
from array import array
from typing import List, Literal
from unittest.mock import patch

import pytest
import torch
from pydantic import BaseModel

from tests.anyscale.json_constrained_decoding.utils import (
    FaultyProcessorWithException)
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, SamplingParams,
                           SequenceData, SequenceGroupMetadata)
from vllm.transformers_utils.tokenizer import get_tokenizer

from vllm.anyscale.constrained_decoding.json_mode_manager import (
    JSONModeManager)
from vllm.anyscale.exceptions import RequestFailedError

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


def create_seq_group_metadata_list(model_id: str,
                                   bsize: int) -> List[SequenceGroupMetadata]:
    """Prepare a list of SequenceGroupMetadata that can be passed to the
        logit processor.

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
        A list of sequence group metadata that can be passed to the logit
        processor.
    """
    seed = 100
    tokenizer = get_tokenizer(model_id)
    random.seed(seed)

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

    seq_group_metadata_list = []
    for i in range(bsize):
        output_ids = prev_output_tokens[i]
        prompt_ids = prompts[i]
        sampling_param = SamplingParams(
            temperature=0.0,
            response_format={
                "type": "json",
                "schema": AnswerFormat.schema_json()
            },
        )
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={
                    0:
                    SequenceData(
                        array(VLLM_TOKEN_ID_ARRAY_TYPE, prompt_ids),
                        _output_token_ids=array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                                output_ids),
                    )
                },
                sampling_params=sampling_param,
                block_tables={0: [1]},
            ))
    return seq_group_metadata_list


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

        manager = JSONModeManager(
            tokenizer_name_or_path=MODEL_ID,
            vocab_size=VOCAB_SIZE,
            recreate_failed_actors=True,
            max_restarts=-1,
            delay_between_actor_restarts_s=0.0,
            logit_processor_cls=FaultyProcessorWithException,
        )

        logits = torch.rand(bsize, VOCAB_SIZE)
        seq_group_metadata_list = create_seq_group_metadata_list(
            MODEL_ID, bsize)

        iteration_cnt = 0
        at_least_one_failure = False
        while iteration_cnt < NUM_FAULT_TOLERANT_ITERATIONS:
            print(f"Iteration {iteration_cnt} ...")
            buffer_inds_to_batch_inds = manager.start_json_logits_processors(
                seq_group_metadata_list)
            _, valid_mask = manager.get_json_logits_bias(
                logits,
                buffer_inds_to_batch_inds,
            )

            if valid_mask is not None and sum(valid_mask) != len(valid_mask):
                at_least_one_failure = True
            iteration_cnt += 1

        assert at_least_one_failure, (
            "At least one failure should have occurred.")


@pytest.mark.asyncio
async def test_json_mode_manager_raise_exception(monkeypatch):
    """Verify failure to obtain json mode mask raises an exception.
    """
    monkeypatch.setenv("ANYSCALE_VLLM_ENABLE_JSON_MODE", "1")
    monkeypatch.setenv(
        "ANYSCALE_VLLM_LOGIT_PROCESSOR_CLS",
        "tests.anyscale.json_constrained_decoding.utils"
        ".FaultyProcessorWithErrorsAllTheTime")
    monkeypatch.setenv("ANYSCALE_VLLM_RECREATE_FAILED_ACTORS", "1")
    monkeypatch.setenv("ANYSCALE_VLLM_DELAY_BETWEEN_ACTOR_RESTARTS_S", "0.0")
    monkeypatch.setenv("ANYSCALE_VLLM_MAX_RESTARTS", "0")
    monkeypatch.setenv("ANYSCALE_VLLM_USE_V2", "0")

    engine_args = AsyncEngineArgs(model=MODEL_ID)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompts = [
        "What is your name? My name is John. I am an engineer. "
        "I like reading,"
        "swimming and hiking. I am 25 years old.",
        "What is your age? My name is Sarah, I am 30 years old. "
        "I am a doctor."
        "I like playing tennis and reading.",
    ]
    stream = await engine.add_request(
        request_id="0",
        inputs=prompts[0],
        params=SamplingParams(
            temperature=0.0,
            response_format={
                "type": "json",
                "schema": AnswerFormat.schema_json()
            },
            stop=["</s>"],
            max_tokens=256,
        ),
    )

    with pytest.raises(RequestFailedError):
        async for request_output in stream:
            print(request_output)


@pytest.mark.asyncio
async def test_json_mode_manager_raise_exception_failure_only(monkeypatch):
    """Verify only a failed seq group raises an exception.

    ProcessorFailFirstRank fails requests coming to the first rank
    logit processor. The test batches 2 requests, and the first
    request is expected to fail because it is sent to the first rank
    logit processor. The test relies on that batch is sent to different
    logit processors in an ascending order.
    """
    monkeypatch.setenv("ANYSCALE_VLLM_ENABLE_JSON_MODE", "1")
    monkeypatch.setenv(
        "ANYSCALE_VLLM_LOGIT_PROCESSOR_CLS",
        "tests.anyscale.json_constrained_decoding.utils"
        ".ProcessorFailFirstRank")
    monkeypatch.setenv("ANYSCALE_VLLM_RECREATE_FAILED_ACTORS", "1")
    monkeypatch.setenv("ANYSCALE_VLLM_DELAY_BETWEEN_ACTOR_RESTARTS_S", "0.0")
    monkeypatch.setenv("ANYSCALE_VLLM_MAX_RESTARTS", "0")
    monkeypatch.setenv("ANYSCALE_VLLM_NUM_PROCESSOR_WORKERS", "2")
    monkeypatch.setenv("ANYSCALE_VLLM_USE_V2", "0")

    engine_args = AsyncEngineArgs(model=MODEL_ID)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompts = [
        "What is your name? My name is John. I am an engineer. "
        "I like reading,"
        "swimming and hiking. I am 25 years old.",
        "What is your age? My name is Sarah, I am 30 years old. "
        "I am a doctor."
        "I like playing tennis and reading.",
    ]

    # Batch 2 requests.
    first_req = engine.add_request(
        request_id="1",
        inputs=prompts[0],
        params=SamplingParams(
            temperature=0.0,
            response_format={
                "type": "json",
                "schema": AnswerFormat.schema_json()
            },
            stop=["</s>"],
            max_tokens=256,
        ),
    )
    second_req = engine.add_request(
        request_id="2",
        inputs=prompts[1],
        params=SamplingParams(
            temperature=0.0,
            response_format={
                "type": "json",
                "schema": AnswerFormat.schema_json()
            },
            stop=["</s>"],
            max_tokens=256,
        ),
    )
    stream1, stream2 = await asyncio.gather(first_req, second_req)
    # First request should fail because it sends the request to the
    # first actor that is supposed to fail ProcessorFailFirstRank.
    with pytest.raises(RequestFailedError):
        async for request_output in stream1:
            assert request_output.error is None

    # Second request should succeeds.
    async for request_output in stream2:
        print(request_output)


def test_engine_e2e_fault_tolerance(monkeypatch):
    """Tests end-to-end fault tolerance of the engine.

    Send two requests to an engine with a faulty logit processor. Run through
    the decoding, at some point the requests will fail and the engine should
    recover and continue without interruption. There should be at least one
    request that receives an exception object during the process.
    """
    monkeypatch.setenv("ANYSCALE_VLLM_ENABLE_JSON_MODE", "1")
    monkeypatch.setenv(
        "ANYSCALE_VLLM_LOGIT_PROCESSOR_CLS",
        "tests.anyscale.json_constrained_decoding.utils"
        ".FaultyJsonModeLogitsProcessor")
    monkeypatch.setenv("ANYSCALE_VLLM_RECREATE_FAILED_ACTORS", "1")
    monkeypatch.setenv("ANYSCALE_VLLM_DELAY_BETWEEN_ACTOR_RESTARTS_S", "0.0")
    monkeypatch.setenv("ANYSCALE_VLLM_MAX_RESTARTS", "-1")

    engine_args = EngineArgs(model=MODEL_ID)
    engine = LLMEngine.from_engine_args(engine_args)
    prompts = [
        "What is your name? My name is John. I am an engineer. "
        "I like reading,"
        "swimming and hiking. I am 25 years old.",
        "What is your age? My name is Sarah, I am 30 years old. "
        "I am a doctor."
        "I like playing tennis and reading.",
    ]
    engine.add_request(
        request_id="0",
        inputs=prompts[0],
        params=SamplingParams(
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
        inputs=prompts[1],
        params=SamplingParams(
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
            if isinstance(output.error, Exception):
                at_least_one_failure = True
                continue
    assert at_least_one_failure, "At least one failure should have occurred."


def test_logit_processor_config_with_str_input():
    """Tests that the config can be init with a str input."""
    logit_processor_cls = ("tests.anyscale.json_constrained_decoding."
                           "utils.FaultyProcessorWithException")
    manager = JSONModeManager(
        tokenizer_name_or_path=MODEL_ID,
        vocab_size=VOCAB_SIZE,
        recreate_failed_actors=True,
        max_restarts=-1,
        delay_between_actor_restarts_s=0.0,
        logit_processor_cls=logit_processor_cls,
    )
    assert manager.logit_processor_cls == FaultyProcessorWithException
