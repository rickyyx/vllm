"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


class LlamaForCausalLM(nn.Module):
    """This is a llama model specific to ScratchLLM. ScratchLLM has its own
    weight loading code, but a subset of weights are supposed to be loaded
    from HF (The final RMS norm and lm head for logit computation). This
    module is used to load those weights.
    """

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        # TODO(sang): Probably not going to work with tp > 1
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      org_num_embeddings=config.vocab_size,
                                      padding_size=DEFAULT_VOCAB_PADDING_SIZE)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.tie_word_embeddings:
            embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )

            self.lm_head.weight = embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError("It is just to load weights.")

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name == "model.norm.weight":
                name = "norm.weight"

            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
