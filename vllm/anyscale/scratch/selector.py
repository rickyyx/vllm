import enum
import os
from collections import namedtuple
from typing import Optional

import torch

from vllm.platforms import current_platform

from vllm.anyscale.scratch.constants import (ROOT_DIRECTORY,
                                             SCRATCH_BUILD_TYPE,
                                             SCRATCH_EXECUTABLE_PATH_ENV_VAR)

ModelProperties = namedtuple(
    "ModelProperties",
    ["arch_type", "shard_size", "device_name", "model_type"])


class SupportedGPUTypes(enum.Enum):
    A10G = enum.auto()
    H100 = enum.auto()


def get_gpu_type() -> SupportedGPUTypes:
    assert current_platform.is_cuda()
    device_name = torch.cuda.get_device_name()
    for name, member in SupportedGPUTypes.__members__.items():
        if name.lower() in device_name.lower():
            return member
    else:
        raise ValueError(f"{device_name} is not supported by Scratch.")


def get_model_properties(model_name: str) -> ModelProperties:
    model_name = model_name.lower()
    arch_type: Optional[str] = None
    shard_size: Optional[int] = None
    device_name: Optional[str] = None
    model_type: Optional[str] = None

    if "llama-2" in model_name:
        arch_type = "ll27b"
    elif "llama-3" in model_name:
        arch_type = "ll38b"
    else:
        raise AssertionError(f"{model_name} is not supported by Scratch")

    if model_name == "meta-llama/llama-2-7b-hf":
        model_type = "ll27b"
    elif model_name == "meta-llama/meta-llama-3-8b":
        model_type = "ll38b"
    elif model_name == "meta-llama/meta-llama-3-8b-instruct":
        model_type = "ll38b-inst"
    else:
        raise AssertionError(f"{model_name} is not supported by Scratch")

    gpu_type = get_gpu_type()
    if gpu_type == SupportedGPUTypes.A10G:
        device_name = "a10g"
        shard_size = 1
    elif gpu_type == SupportedGPUTypes.H100:
        device_name = "h100"
        shard_size = 4

    assert arch_type is not None
    assert shard_size is not None
    assert device_name is not None
    assert model_type is not None

    return ModelProperties(
        arch_type=arch_type,
        shard_size=shard_size,
        device_name=device_name,
        model_type=model_type,
    )


def get_scratch_executable_path(model_name: str) -> str:
    # Executable looks like ll38b-s4-h100-f16
    properties = get_model_properties(model_name)
    executable_prefix = (f"{properties.arch_type}-s{properties.shard_size}"
                         f"-{properties.device_name}-f16")

    return os.getenv(
        SCRATCH_EXECUTABLE_PATH_ENV_VAR,
        f"{ROOT_DIRECTORY}/scratch-{executable_prefix}-{SCRATCH_BUILD_TYPE}"
        ".cpython-39-x86_64-linux-gnu.so")


def get_scratch_weights_uri(model_name: str) -> str:
    # Weights looks like ll38b-inst-s4-h100-f16
    properties = get_model_properties(model_name)
    weights_path_prefix = (f"{properties.model_type}-s{properties.shard_size}"
                           f"-{properties.device_name}-f16")
    return f"staging_weights/{weights_path_prefix}"
