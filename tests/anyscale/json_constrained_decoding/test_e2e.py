# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.

import json
import os
from typing import List

import pytest
from pydantic import BaseModel, Field

from vllm.sequence import SamplingParams

os.environ["ANYSCALE_VLLM_ENABLE_JSON_MODE"] = "1"


class BasicResponse(BaseModel):
    """The format of the answer."""

    winner_team_name: str = Field(description="Name of the winning team")
    loser_team_name: str = Field(description="Name of the losing team")
    winner_score: int = Field(description="Score of the winning team")
    loser_score: int = Field(description="Score of the losing team")


class ArrayResponse(BaseModel):
    """The format of the answer."""

    sorted_numbers: List[int] = Field(description="List of the sorted numbers")


class Person(BaseModel):
    """The object representing a person with name and age"""

    name: str = Field(description="Name of the person")
    age: int = Field(description="The age of the person")


class NestedResponse(BaseModel):
    """The format of the answer."""

    sorted_list: List[Person] = Field(description="List of the sorted objects")


AnyOfSchema = {
    "properties": {
        "groceryList": {
            "items": {
                "anyOf": [
                    {
                        "title": "Produce",
                        "type": "string"
                    },
                    {
                        "title": "Dairy",
                        "type": "string"
                    },
                ]
            },
            "title": "GroceryList",
            "type": "array"
        }
    },
    "required": ["groceryList"],
    "title": "GroceryListResponse",
    "type": "object"
}

AnyOfSchemaNumber = {
    "properties": {
        "groceryList": {
            "items": {
                "anyOf": [
                    {
                        "title": "Produce",
                        "type": "string"
                    },
                    {
                        "title": "Dairy",
                        "type": "string"
                    },
                    {
                        "title": "Foo",
                        "type": "number"
                    },
                ]
            },
            "title": "GroceryList",
            "type": "array"
        }
    },
    "required": ["groceryList"],
    "title": "GroceryListResponse",
    "type": "object"
}

response_types = ["basic", "array", "nested", "anyof", "anyof_number"]


def get_prompt_and_expected_type(response_type: str):
    system_prompt = "You are a helpful assistant designed to output JSON."
    if response_type == "basic":
        prompt = (
            f"[INST] {system_prompt} + Who won the world series in 2020? "
            "[/INST]")
        expected_type = BasicResponse
    elif response_type == "array":
        prompt = (f"[INST] {system_prompt} + Sort the numbers 3, 1, 2, 4, 5 "
                  "[/INST]")
        expected_type = ArrayResponse
    elif response_type == "nested":
        prompt = (
            f"[INST] {system_prompt} + Sort these people by age: John, "
            "20 years old, Mary, 30 years old, Bob, 10 years old. [/INST]")
        expected_type = NestedResponse
    elif response_type == "anyof":
        prompt = (
            f"[INST] {system_prompt} + Create a grocery list for a dinner "
            "party with 8 people [/INST]")
        expected_type = AnyOfSchema
    elif response_type == "anyof_number":
        prompt = (
            f"[INST] {system_prompt} + Create a grocery list for a dinner "
            "party with 8 people [/INST]")
        expected_type = AnyOfSchemaNumber
    else:
        raise ValueError(
            (f"Unknown response type {response_type} only basic, array, and "
             "nested are supported"))

    return prompt, expected_type


def get_response_formats():
    return [
        # TODO (Kourosh): The following should be supported
        {
            "type": "json_object"
        },
        {
            "type": "json_object",
            "schema": {}
        },
        {
            "type": "json_object",
            "schema": json.dumps({})
        },
        {
            "type": "json_object",
            "schema": json.loads(BasicResponse.schema_json())
        },
        {
            "type": "json_object",
            "schema": BasicResponse.schema_json()
        },
    ]


# TODO(sang): Enable v2
# @pytest.mark.parametrize("json_mode_version", ["v1", "v2"])
@pytest.mark.parametrize("json_mode_version", ["v1"])
def test_json_mode(mistral_7b_json):
    """Check if we can run without exceptions and that output is valid."""
    engine = mistral_7b_json

    test_prompts = []
    expected_types = []

    for response_type in response_types:
        prompt, expected_type = get_prompt_and_expected_type(response_type)

        if isinstance(expected_type, dict):
            schema = json.dumps(expected_type)
        else:
            # It is a base model, so we need to convert it to a schema.
            schema = expected_type.schema_json()

        test_prompts.append((prompt,
                             SamplingParams(temperature=1.0,
                                            response_format={
                                                "type": "json",
                                                "schema": schema
                                            },
                                            max_tokens=512)))
        expected_types.append(expected_type)

    # Repeat 3 times to test consistency.
    for _ in range(3):
        # Run the engine by calling `engine.step()` manually.
        curr_test_prompts = test_prompts.copy()
        request_id = 0
        all_request_outputs = [None] * len(test_prompts)

        while engine.has_unfinished_requests() or curr_test_prompts:
            # To test continuous batching, we add one request at each step.
            if curr_test_prompts:
                prompt, sampling_params = curr_test_prompts.pop(0)
                engine.add_request(str(request_id), prompt, sampling_params)
                request_id += 1

            request_outputs = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    all_request_outputs[int(
                        request_output.request_id)] = request_output

        for output, expected_type in zip(all_request_outputs, expected_types):
            if isinstance(expected_type, type):
                # If it's a pydantic type also load to see if it's correct.
                expected_type(**json.loads(output.outputs[0].text))
            else:
                json.loads(output.outputs[0].text)
