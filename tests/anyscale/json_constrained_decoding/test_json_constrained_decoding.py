# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

import json
# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.
import os
from typing import List

import pytest
from jsonschema import validate
from pydantic import BaseModel, Field

from vllm import SamplingParams

os.environ["ANYSCALE_VLLM_ENABLE_JSON_MODE"] = "1"

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
@pytest.mark.parametrize("json_mode_version", ["v1", "v2"])
def test_json_constrained_decoding(
    monkeypatch,
    vllm_runner,
    model: str,
    schema: str,
    json_mode_version: str,
) -> None:
    """Test that the model can generate a valid answer for a json schema."""
    if json_mode_version == "v2":
        monkeypatch.setenv("ANYSCALE_VLLM_USE_V2", "1")

    print(f"loading model {model}...")
    model = vllm_runner(
        model,
        max_model_len=4096,
    )

    response_format = {"type": "json", "schema": schema}
    sampling_params = SamplingParams(response_format=response_format,
                                     max_tokens=200,
                                     temperature=0.0)
    prompt = (f"Say something in the following JSON {schema}")
    outs = model.generate(prompts=[prompt], sampling_params=sampling_params)
    ans = json.loads(outs[0][1][0][len(prompt):])  # codespell:ignore
    validate(instance=ans, schema=json.loads(schema))  # codespell:ignore
