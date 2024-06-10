# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.

import huggingface_hub
import pytest
import torch.nn as nn

import vllm

from .utils import SecretManager


@pytest.fixture(scope="session", autouse=True)
def load_secrets():
    secrets = SecretManager()
    token = secrets.override_secret("HUGGING_FACE_HUB_TOKEN")
    huggingface_hub.login(token)


@pytest.fixture
def mistral_7b_json(
    load_secrets,  # pylint: disable=unused-argument, redefined-outer-name
    json_mode_version,
) -> nn.Module:
    # cleanup()

    enable_json_mode_v2 = (json_mode_version == "v2")
    engine = vllm.LLM("mistralai/Mistral-7B-Instruct-v0.1",
                      worker_use_ray=True,
                      enable_lora=False,
                      enable_json_logits_processors=True,
                      max_model_len=8192,
                      enable_json_mode_v2=enable_json_mode_v2)
    yield engine.llm_engine
    del engine
    # cleanup()
