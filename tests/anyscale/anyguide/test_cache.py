import gc

import numpy as np
import pytest
from anyguide import PrefixCache, Stack, Trie

from .utils import create_element

VALUE = (np.ones(10, dtype=np.bool_), [Trie({"foo": 1})])


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
        local_prefix_cache[stack] = {tokenizer_trie: VALUE}
        assert local_prefix_cache[stack] == {tokenizer_trie: VALUE}

    def test_get_item_prefix_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack, tokenizer_trie] = VALUE
        assert local_prefix_cache[stack, tokenizer_trie] == VALUE

    def test_set_item_single_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack] = {tokenizer_trie: VALUE}
        assert local_prefix_cache[stack] == {tokenizer_trie: VALUE}

    def test_set_item_prefix_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack, tokenizer_trie] = VALUE
        assert local_prefix_cache[stack, tokenizer_trie] == VALUE

    def test_contains_single_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack] = {tokenizer_trie: VALUE}
        assert stack in local_prefix_cache

    def test_contains_prefix_key(self, local_prefix_cache, stack,
                                 tokenizer_trie):
        local_prefix_cache[stack, tokenizer_trie] = VALUE
        assert (stack, tokenizer_trie) in local_prefix_cache

    def test_contains_non_existing_key(self, local_prefix_cache, stack,
                                       tokenizer_trie):
        assert (stack, tokenizer_trie) not in local_prefix_cache

    def test_get_item_non_existing_single_key(self, local_prefix_cache, stack):
        with pytest.raises(KeyError):
            local_prefix_cache[stack]  # pylint: disable=pointless-statement

    # def test_prefix_destructor(self):
    #     prefix_cache = PrefixCache()
    #     trie = Trie({"foo": 1})
    #     prefix_cache[Stack()] = {trie: VALUE}
    #     # prefix_cache[Stack(), trie] = VALUE
    #     breakpoint()

    # def test_set_item_invalid_key(self, local_prefix_cache, tokenizer_trie):
    #     # Test setting item with invalid key
    #     with pytest.raises(ValueError):
    #         local_prefix_cache["invalid_key"] = {tokenizer_trie: VALUE}

    # def test_set_item_invalid_value(self, local_prefix_cache, stack):

    #     with pytest.raises(ValueError):
    #         local_prefix_cache[stack] = VALUE

    def test_prefix_cache_tuple_key(self):

        def fn():
            prefix_cache = PrefixCache()
            trie1 = Trie({"foo": 1})
            trie2 = Trie({"bar": 2})

            mask = [True, False]
            value = (mask, [trie1, trie2])

            stack = Stack()
            prefix_cache[stack, trie1] = value

            # Retrieve the stored value and check its contents
            retrieved_value = prefix_cache[stack, trie1]
            assert retrieved_value[0].tolist() == mask

            # Check if the retrieved Trie objects are equal to the original ones
            assert retrieved_value[1][0] == trie1
            assert retrieved_value[1][1] == trie2

            # Explicitly delete the Trie objects to trigger the
            # heap-use-after-free error
            del trie1
            del trie2

            # Attempt to access the stored value again
            # If the heap-use-after-free error occurs, this line will raise an
            # exception
            retrieved_value = prefix_cache[stack, retrieved_value[1][0]]

            # If no exception is raised, the test passes
            assert True

        fn()
        gc.collect()
        assert True
