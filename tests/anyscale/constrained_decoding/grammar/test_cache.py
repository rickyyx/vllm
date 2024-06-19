import pytest

from vllm.anyscale.constrained_decoding.grammar.cache import PrefixCache
from vllm.anyscale.constrained_decoding.grammar.element import (Element,
                                                                ElementType)
from vllm.anyscale.constrained_decoding.grammar.stack import Stack


def create_element(etype=ElementType.CHAR,
                   code_point=None,
                   upper_bound=None,
                   alternatives=None,
                   rule_name=None):
    """A helper function to create an Element object quickly"""
    return Element(etype=etype,
                   code_point=code_point,
                   upper_bound=upper_bound,
                   alternatives=alternatives,
                   rule_name=rule_name).freeze()


class TestPrefixCache:

    # Define fixtures if needed
    @pytest.fixture(scope="function")
    def local_prefix_cache(self):
        return PrefixCache()

    @pytest.fixture(scope="module")
    def stack(self):
        elements = [
            create_element(code_point=ord("a")),
            create_element(code_point=ord("b")),
            create_element(code_point=ord("c"))
        ]
        return Stack(elements)

    def test_get_item_single_key(self, local_prefix_cache, tokenizer_trie,
                                 stack):
        local_prefix_cache[stack] = {tokenizer_trie: "value1"}
        assert local_prefix_cache[stack] == {tokenizer_trie: "value1"}

    def test_get_item_prefix_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack, tokenizer_trie] = "value1"
        assert local_prefix_cache[stack, tokenizer_trie] == "value1"

    def test_set_item_single_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack] = {tokenizer_trie: "value1"}
        assert local_prefix_cache[stack] == {tokenizer_trie: "value1"}

    def test_set_item_prefix_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack, tokenizer_trie] = "value1"
        assert local_prefix_cache[stack, tokenizer_trie] == "value1"

    def test_contains_single_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack] = {tokenizer_trie: "value1"}
        assert stack in local_prefix_cache

    def test_contains_prefix_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack, tokenizer_trie] = "value1"
        assert (stack, tokenizer_trie) in local_prefix_cache

    def test_contains_non_existing_key(self, local_prefix_cache, stack,
                                       tokenizer_trie):
        assert (stack, tokenizer_trie) not in local_prefix_cache

    def test_get_item_non_existing_single_key(self, local_prefix_cache, stack):
        with pytest.raises(KeyError):
            local_prefix_cache[stack]  # pylint: disable=pointless-statement

    def test_set_item_invalid_key(self, local_prefix_cache):
        # Test setting item with invalid key
        with pytest.raises(ValueError):
            local_prefix_cache["invalid_key"] = "value"

    def test_set_item_invalid_value(self, local_prefix_cache, stack):

        with pytest.raises(ValueError):
            local_prefix_cache[stack] = "invalid_value"
