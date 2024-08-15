# ruff: noqa
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.anyscale.constrained_decoding.grammar.cache import PrefixCache
from vllm.anyscale.constrained_decoding.grammar.json_cache import (
    compute_json_mode_cache)
from vllm.anyscale.constrained_decoding.grammar.trie import TokenizerTrie


@pytest.fixture(scope="class")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    hf_tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m")
    return hf_tokenizer


@pytest.fixture(scope="class")
def tokenizer_trie(hf_tokenizer: PreTrainedTokenizerBase) -> TokenizerTrie:
    return TokenizerTrie(hf_tokenizer)


@pytest.fixture(scope="class")
def prefix_cache(tokenizer_trie: TokenizerTrie) -> PrefixCache:
    return compute_json_mode_cache(tokenizer_trie)
