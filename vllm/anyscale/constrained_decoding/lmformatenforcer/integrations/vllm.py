# ruff: noqa

try:
    import torch
    from transformers import PreTrainedTokenizerBase

    import vllm
except ImportError:
    raise ImportError(
        'vllm is not installed. Please install it with "pip install vllm"')
import math
from typing import List, Optional, Union

from lmformatenforcer import (CharacterLevelParser, FormatEnforcerAnalyzer,
                              TokenEnforcer, TokenEnforcerTokenizerData)
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data)


class VLLMLogitsProcessor:

    def __init__(self, token_enforcer: TokenEnforcer, analyze):
        self.token_enforcer = token_enforcer
        self.analyzer = FormatEnforcerAnalyzer(
            token_enforcer) if analyze else None
        self.mask: Optional[torch.Tensor] = None

    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        token_sequence = input_ids
        if self.analyzer:
            self.analyzer.report_raw_logits(token_sequence, scores.tolist())
        allowed_tokens = self.token_enforcer.get_allowed_tokens(token_sequence)
        if self.mask is not None:
            self.mask.fill_(-math.inf)
        else:
            # We create it here because full_like() also copies the device and dtype
            self.mask = torch.full_like(scores, -math.inf)
        self.mask[allowed_tokens] = 0
        scores = scores + self.mask
        return scores


def build_vllm_token_enforcer_tokenizer_data(
    llm: Union[vllm.LLM,
               PreTrainedTokenizerBase]) -> TokenEnforcerTokenizerData:
    tokenizer = llm.get_tokenizer() if isinstance(llm, vllm.LLM) else llm
    # In some vLLM versions the tokenizer is wrapped in a TokenizerGroup
    if tokenizer.__class__.__name__ == 'TokenizerGroup':
        tokenizer = tokenizer.tokenizer  # noqa
    return build_token_enforcer_tokenizer_data(tokenizer)


def build_vllm_logits_processor(llm: Union[vllm.LLM, PreTrainedTokenizerBase,
                                           TokenEnforcerTokenizerData],
                                character_level_parser: CharacterLevelParser,
                                analyze: bool = False) -> VLLMLogitsProcessor:
    """Build the logits processor function that llama.cpp will use to filter the tokens generated by the model. The result
    can be passed in the logits_processor list that is sent to the call or generate() method of llama.cpp models."""
    if not isinstance(llm, TokenEnforcerTokenizerData):
        llm = build_vllm_token_enforcer_tokenizer_data(llm)
    token_enforcer = TokenEnforcer(llm, character_level_parser)
    return VLLMLogitsProcessor(token_enforcer, analyze)


__all__ = [
    'build_vllm_logits_processor', 'build_vllm_token_enforcer_tokenizer_data'
]