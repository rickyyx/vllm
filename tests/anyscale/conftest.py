# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.

import contextlib
import gc

import pytest
import torch
import torch.nn as nn

import vllm
from vllm.distributed import destroy_model_parallel


def cleanup():
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_vllm():
    yield
    cleanup()


@pytest.fixture
def mistral_7b_json(json_mode_version, ) -> nn.Module:
    cleanup()

    enable_json_mode_v2 = (json_mode_version == "v2")
    # TODO(sang): Remove it once v2 is supported.
    enable_json_mode_v2 = False
    print(enable_json_mode_v2)
    engine = vllm.LLM(
        "mistralai/Mistral-7B-Instruct-v0.1",
        max_model_len=8192,
    )
    yield engine.llm_engine
    del engine
    cleanup()
