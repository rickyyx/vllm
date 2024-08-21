import functools
import importlib
from typing import Dict, List, Optional, Tuple, Type

import torch.nn as nn

from vllm.logger import init_logger
from vllm.utils import is_hip

from vllm.anyscale.anyscale_envs import USE_SCRATCH

logger = init_logger(__name__)

_GENERATION_MODELS = {
    "AquilaModel": ("llama", "LlamaForCausalLM"),
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),  # AquilaChat2
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),  # baichuan-7b
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),  # baichuan-13b
    "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "CohereForCausalLM": ("commandr", "CohereForCausalLM"),
    "DbrxForCausalLM": ("dbrx", "DbrxForCausalLM"),
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "DeepseekForCausalLM": ("deepseek", "DeepseekForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    "Gemma2ForCausalLM": ("gemma2", "Gemma2ForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "InternLMForCausalLM": ("llama", "LlamaForCausalLM"),
    "InternLM2ForCausalLM": ("internlm2", "InternLM2ForCausalLM"),
    "JAISLMHeadModel": ("jais", "JAISLMHeadModel"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    "QuantMixtralForCausalLM": ("mixtral_quant", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    "MptForCausalLM": ("mpt", "MPTForCausalLM"),
    "MPTForCausalLM": ("mpt", "MPTForCausalLM"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMForCausalLM"),
    "NemotronForCausalLM": ("nemotron", "NemotronForCausalLM"),
    "OlmoForCausalLM": ("olmo", "OlmoForCausalLM"),
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "OrionForCausalLM": ("orion", "OrionForCausalLM"),
    "PersimmonForCausalLM": ("persimmon", "PersimmonForCausalLM"),
    "PhiForCausalLM": ("phi", "PhiForCausalLM"),
    "Phi3ForCausalLM": ("llama", "LlamaForCausalLM"),
    "QWenLMHeadModel": ("qwen", "QWenLMHeadModel"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "RWForCausalLM": ("falcon", "FalconForCausalLM"),
    "StableLMEpochForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    "ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
    "XverseForCausalLM": ("xverse", "XverseForCausalLM"),
    "Phi3SmallForCausalLM": ("phi3_small", "Phi3SmallForCausalLM"),
    "MedusaModel": ("medusa", "Medusa"),
    "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
    "JambaForCausalLM": ("jamba", "JambaForCausalLM")
}

_EMBEDDING_MODELS = {
    "MistralModel": ("llama_embedding", "LlamaEmbeddingModel"),
}

_MULTIMODAL_MODELS = {
    "Blip2ForConditionalGeneration":
    ("blip2", "Blip2ForConditionalGeneration"),
    "ChameleonForConditionalGeneration":
    ("chameleon", "ChameleonForConditionalGeneration"),
    "FuyuForCausalLM": ("fuyu", "FuyuForCausalLM"),
    "InternVLChatModel": ("internvl", "InternVLChatModel"),
    "LlavaForConditionalGeneration":
    ("llava", "LlavaForConditionalGeneration"),
    "LlavaNextForConditionalGeneration":
    ("llava_next", "LlavaNextForConditionalGeneration"),
    "MiniCPMV": ("minicpmv", "MiniCPMV"),
    "PaliGemmaForConditionalGeneration": ("paligemma",
                                          "PaliGemmaForConditionalGeneration"),
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
}
_CONDITIONAL_GENERATION_MODELS = {
    "BartModel": ("bart", "BartForConditionalGeneration"),
    "BartForConditionalGeneration": ("bart", "BartForConditionalGeneration"),
}

_MODELS = {
    **_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_MULTIMODAL_MODELS,
    **_CONDITIONAL_GENERATION_MODELS,
}

# Architecture -> type.
# out of tree models
_OOT_MODELS: Dict[str, Type[nn.Module]] = {}

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS: List[str] = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_SWA_REASON = ("Sliding window attention (SWA) is not yet supported in "
                    "Triton flash attention. For half-precision SWA support, "
                    "please use CK flash attention by setting "
                    "`VLLM_USE_TRITON_FLASH_ATTN=0`")
_ROCM_PARTIALLY_SUPPORTED_MODELS: Dict[str, str] = {
    "Qwen2ForCausalLM":
    _ROCM_SWA_REASON,
    "MistralForCausalLM":
    _ROCM_SWA_REASON,
    "MixtralForCausalLM":
    _ROCM_SWA_REASON,
    "PaliGemmaForConditionalGeneration":
    ("ROCm flash attention does not yet "
     "fully support 32-bit precision on PaliGemma"),
    "Phi3VForCausalLM":
    ("ROCm Triton flash attention may run into compilation errors due to "
     "excessive use of shared memory. If this happens, disable Triton FA "
     "by setting `VLLM_USE_TRITON_FLASH_ATTN=0`")
}


class ModelRegistry:

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _get_model(model_arch: str):
        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"vllm.model_executor.models.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def _try_load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch in _OOT_MODELS:
            return _OOT_MODELS[model_arch]
        if model_arch not in _MODELS:
            return None
        if is_hip():
            if model_arch in _ROCM_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"Model architecture {model_arch} is not supported by "
                    "ROCm for now.")
            if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(
                    "Model architecture %s is partially supported by ROCm: %s",
                    model_arch, _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch])

        return ModelRegistry._get_model(model_arch)

    @staticmethod
    def resolve_model_cls(
            architectures: List[str]) -> Tuple[Type[nn.Module], str]:
        for arch in architectures:
            model_cls = ModelRegistry._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {ModelRegistry.get_supported_archs()}")

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys()) + list(_OOT_MODELS.keys())

    @staticmethod
    def register_model(model_arch: str, model_cls: Type[nn.Module]):
        if model_arch in _MODELS:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.", model_arch,
                model_cls.__name__)
        global _OOT_MODELS
        _OOT_MODELS[model_arch] = model_cls

    @staticmethod
    def is_embedding_model(model_arch: str) -> bool:
        return model_arch in _EMBEDDING_MODELS

    @staticmethod
    def is_multimodal_model(model_arch: str) -> bool:

        # TODO: find a way to avoid initializing CUDA prematurely to
        # use `supports_multimodal` to determine if a model is multimodal
        # model_cls = ModelRegistry._try_load_model_cls(model_arch)
        # from vllm.model_executor.models.interfaces import supports_multimodal
        return model_arch in _MULTIMODAL_MODELS


# Anyscale start
_SCRATCH_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
}


class ScratchModelRegistry:
    """Model registry that can load ScratchLLM models.

    The API is compatible to ModelRegistry.
    """

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_SCRATCH_MODELS.keys())

    @staticmethod
    def _load_scratch_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        assert USE_SCRATCH
        if model_arch not in _SCRATCH_MODELS:
            raise ValueError(f"{model_arch} is not supported for ScratchLLM.")

        module_name, model_cls_name = _SCRATCH_MODELS[model_arch]
        module = importlib.import_module(
            f"vllm.anyscale.scratch.models.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def register_model(model_arch: str, model_cls: Type[nn.Module]):
        raise NotImplementedError(
            "Scratch model loader doesn't support register_model")

    @staticmethod
    def is_embedding_model(model_arch: str) -> bool:
        return False

    @staticmethod
    def is_multimodal_model(model_arch: str) -> bool:
        return False

    @staticmethod
    def resolve_model_cls(
            architectures: List[str]) -> Tuple[Type[nn.Module], str]:
        for arch in ScratchModelRegistry.get_supported_archs():
            model_cls = ScratchModelRegistry._load_scratch_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            "Supported architectures: "
            f"{ScratchModelRegistry.get_supported_archs()}")


if USE_SCRATCH:
    OriginalModelRegistry = ModelRegistry
    ModelRegistry = ScratchModelRegistry

# Anyscale end

__all__ = [
    "ModelRegistry",
]
