import enum
from collections import namedtuple
from pathlib import Path
from typing import Optional

import torch
from scratchllm import ScratchDType, ScratchHardware

from vllm.platforms import current_platform

from vllm.anyscale import anyscale_envs
from vllm.anyscale.scratch.constants import (SCRATCH_TMP_DIR,
                                             SCRATCH_WEIGHTS_BUCKET_NAME,
                                             SCRATCH_WEIGHTS_PATH_PREFIX,
                                             SCRATCH_WEIGHTS_VERSION_COMMIT)

ModelProperties = namedtuple("ModelProperties",
                             ["arch_type", "device_name", "model_type"])

TORCH_DTYPE_TO_SCRATCH_DTYPE = {
    torch.half: ScratchDType.HALF,
    torch.bfloat16: ScratchDType.BFLOAT16,
}


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
    if anyscale_envs.SCRATCH_HF_MODEL_ID:
        model_name = anyscale_envs.SCRATCH_HF_MODEL_ID

    model_name = model_name.lower()
    arch_type: Optional[str] = None
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
    elif model_name == "meta-llama/llama-2-7b-chat-hf":
        model_type = "ll27b-chat"
    elif model_name == "meta-llama/meta-llama-3-8b":
        model_type = "ll38b"
    elif model_name == "meta-llama/meta-llama-3-8b-instruct":
        model_type = "ll38b-inst"
    else:
        raise AssertionError(f"{model_name} is not supported by Scratch")

    gpu_type = get_gpu_type()
    if gpu_type == SupportedGPUTypes.A10G:
        device_name = "a10g"
    elif gpu_type == SupportedGPUTypes.H100:
        device_name = "h100"

    assert arch_type is not None
    assert device_name is not None
    assert model_type is not None

    return ModelProperties(
        arch_type=arch_type,
        device_name=device_name,
        model_type=model_type,
    )


def get_scratch_dtype(torch_dtype: torch.dtype) -> ScratchDType:
    return TORCH_DTYPE_TO_SCRATCH_DTYPE[torch_dtype]


def get_scratch_hardware() -> ScratchHardware:
    gpu_type = get_gpu_type()
    if gpu_type == SupportedGPUTypes.A10G:
        return ScratchHardware.A10G
    elif gpu_type == SupportedGPUTypes.H100:
        return ScratchHardware.H100
    else:
        raise AssertionError(f"{gpu_type} is not supported by Scratch")


def get_scratch_tmp_dir(model_name: str) -> Path:
    properties = get_model_properties(model_name)
    return Path(SCRATCH_TMP_DIR) / f"{properties.model_type}"


def get_scratch_weights_uri(model_name: str, dtype: torch.dtype) -> str:
    # Weights looks like ll38b-inst-h100-f16
    properties = get_model_properties(model_name)
    dtype = get_scratch_dtype(dtype)
    # Replace / with -- in model name.
    model_name = model_name.replace("/", "--")
    weights_path_prefix = f"{model_name}-{properties.device_name}-{dtype}"
    return (f"{SCRATCH_WEIGHTS_BUCKET_NAME}/"
            f"{SCRATCH_WEIGHTS_PATH_PREFIX}/"
            f"{SCRATCH_WEIGHTS_VERSION_COMMIT}/"
            f"{weights_path_prefix}")
