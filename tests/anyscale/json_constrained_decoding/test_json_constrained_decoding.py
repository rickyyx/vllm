# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.
import gc
import json
from typing import List

import pytest
import ray
import torch
from jsonschema import validate
from pydantic import BaseModel, Field

from vllm import SamplingParams

# Mistral is the model we actually use for function calling.
# facebook/opt-125m still gives results that are correctly constrained
# but I can't find a seed that gives answers expected by test cases.
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
]


class Answer(BaseModel):
    answer: int


class AnswerList(BaseModel):
    answers: List[int]


class AnswerNested(BaseModel):
    answer: Answer


class BasicResponse(BaseModel):
    """The format of the answer."""

    winner_team_name: str = Field(description="Name of the winning team")
    loser_team_name: str = Field(description="Name of the losing team")
    winner_score: int = Field(description="Score of the winning team")
    loser_score: int = Field(description="Score of the losing team")


def get_schema_variations() -> List[str]:
    pydantic_objects = [AnswerList, Answer, AnswerNested]
    pydantic_schemas = [obj.schema_json() for obj in pydantic_objects]
    extra_schemas = [
        json.dumps({
            "type": "object",
            "properties": {
                "street_name": {
                    "type": "string"
                },
            },
            "required": ["street_name"],
        }),
    ]

    return pydantic_schemas + extra_schemas


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("schema", get_schema_variations())
# NOTE: Turned off fast mode tests as v2 is a replacement (for now)
@pytest.mark.parametrize("fast_json_mode", [False])
@pytest.mark.parametrize("json_mode_version", ["v1", "v2"])
def test_json_constrained_decoding(
    vllm_runner,
    model: str,
    schema: str,
    fast_json_mode: bool,
    json_mode_version: str,
) -> None:
    """Test that the model can generate a valid answer for a json schema."""

    if json_mode_version == "v2" and fast_json_mode:
        pytest.skip("json_mode_version v2 is already fast")

    print(f"loading model {model}...")
    model = vllm_runner(
        model,
        max_model_len=4096,
        enable_json_logits_processors=True,
        worker_use_ray=False,
        enable_json_processor_fast_mode=fast_json_mode,
        enable_json_mode_v2=(json_mode_version == "v2"),
    )

    response_format = {"type": "json", "schema": schema}
    sampling_params = SamplingParams(response_format=response_format,
                                     max_tokens=200,
                                     temperature=0.0)
    prompt = (f"Say something in the following JSON {schema}")
    outs = model.generate(prompts=[prompt],
                          sampling_params=sampling_params,
                          return_output_only=True)
    print(outs[0][1][0])
    ans = json.loads(outs[0][1][0])  # codespell:ignore
    validate(instance=ans, schema=json.loads(schema))  # codespell:ignore


# TODO (Kourosh): Remove fast mode tests as v2 is a replacement (for now)
@pytest.mark.parametrize("model_id", MODELS)
def test_fast_json_processor_string_response(
    vllm_runner,
    model_id: str,
):
    """Test legibility of string responses for json constrained decoding.

    Generating strings is strictly harder than generating other simple types
    like integers, so we test it separately.
    """
    answer_type = BasicResponse
    json_schema = answer_type.schema_json()
    response_format = {"type": "json", "schema": json_schema}
    sampling_params = SamplingParams(response_format=response_format,
                                     max_tokens=200,
                                     temperature=0.0)
    prompt = (
        "Who won the world series in 2020?. answer in the following json "
        f"schema: {json_schema}")

    print(f"loading model {model_id}...")
    model = vllm_runner(
        model_id,
        max_model_len=4096,
        enable_json_logits_processors=True,
        enable_json_processor_fast_mode=False,
        worker_use_ray=False,
    )

    outs = model.generate(prompts=[prompt],
                          sampling_params=sampling_params,
                          return_output_only=True)
    ans_not_fast_mode = json.loads(outs[0][1][0])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    model_fast_json = vllm_runner(
        model_id,
        max_model_len=4096,
        enable_json_logits_processors=True,
        enable_json_processor_fast_mode=True,
        worker_use_ray=False,
    )
    outs_fast_mode = model_fast_json.generate(prompts=[prompt],
                                              sampling_params=sampling_params,
                                              return_output_only=True)
    ans_fast_mode = json.loads(outs_fast_mode[0][1][0])
    assert ans_not_fast_mode == ans_fast_mode
    validate(instance=ans_fast_mode,
             schema=json.loads(answer_type.schema_json()))
    validate(instance=ans_not_fast_mode,
             schema=json.loads(answer_type.schema_json()))
