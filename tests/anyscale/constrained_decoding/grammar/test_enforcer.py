import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from .utils import (gen_all_json_schema_example_pairs, gen_all_json_schemas,
                    print_stats, set_seed)
from vllm.anyscale.constrained_decoding.grammar.cache import PrefixCache
from vllm.anyscale.constrained_decoding.grammar.enforcer import GrammarEnforcer
from vllm.anyscale.constrained_decoding.grammar.json_schema_to_grammar import (
    HEX_DIGIT, SPACE_END, SPACE_RULE, STRING)
from vllm.anyscale.constrained_decoding.grammar.parser import Grammar
from vllm.anyscale.constrained_decoding.grammar.trie import TokenizerTrie

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

EXCLUDED_PERFORMANCE_TESTS = {"string-basic"}


class TestGrammarEnforcerAcceptance:
    """Tests that the GrammarEnforcer can tell if a generated response is valid.

    This is done by consecutively calling `accept_char` for each character in
    the response.
    """

    @classmethod
    def assert_accepts(cls,
                       grammar_enforcer: GrammarEnforcer,
                       response: str,
                       expect_error: bool = False):
        """Asserts that the response is accepted by the grammar enforcer.

        This uses `accept_char` to check each character in the response.
        """

        grammar_enforcer.init()
        try:
            for c in response:
                assert grammar_enforcer.accept_char(c)
        except AssertionError:
            if expect_error:
                return
            raise
        else:
            if expect_error:
                raise AssertionError("Expected error, but none was raised.")

    @pytest.mark.parametrize("test_name, schema, example, is_valid",
                             gen_all_json_schema_example_pairs())
    def test_accept_char_extensive(self, tokenizer_trie: TokenizerTrie,
                                   test_name, schema, example, is_valid):

        grammar = Grammar.from_json_schema(schema)
        ge = GrammarEnforcer(grammar=grammar, tokenizer_trie=tokenizer_trie)

        print(f"[{test_name}] Testing\n`{example}`\nwith schema\n`{schema}")
        self.assert_accepts(grammar_enforcer=ge,
                            response=example,
                            expect_error=not is_valid)

    def _run_random_sampling(self,
                             *,
                             schema: str,
                             tokenizer_trie: TokenizerTrie,
                             prefix_cache: PrefixCache,
                             max_length: int = 1000,
                             verbose: bool = False):
        """Runs random sampling with the given schema and tokenizer trie.

        Args:
            schema: The schema to use.
            tokenizer_trie: The tokenizer trie to use.
            prefix_cache: The prefix cache to use.
            max_length: The maximum length of the generated response.
            verbose: Whether to print the allowed tokens at each step.

        Returns:
            A tuple:
            - A list of overheads for each iteration.
            - A list of masks for each iteration.
            - A list of token ids for each iteration.
        """

        grammar = Grammar.from_json_schema(schema)
        ge = GrammarEnforcer(
            grammar=grammar,
            tokenizer_trie=tokenizer_trie,
            global_stack_prefix_cache=prefix_cache,
        )
        ge.init()

        set_seed(42)
        token_id = None
        overheads = []
        masks = []
        token_ids = []
        print("Generating tokens:\n")
        for i in range(max_length):
            s = time.time()
            if token_id is not None:
                assert ge.accept_token(token_id)

            mask = ge.get_tokens_mask()
            masks.append(mask)
            overhead = time.time() - s
            overheads.append(overhead)

            if verbose:
                print("\n")
                allowed_tokens = [
                    tokenizer_trie.id2str(int(t)) for t in np.where(mask)[0]
                ]
                print(f"Allowed tokens ({len(allowed_tokens)}) "
                      f"top_10: {allowed_tokens[:10]}")

            probs = torch.as_tensor(mask / mask.sum())
            token_id = int(torch.multinomial(probs, 1).item())
            token_ids.append(token_id)

            token = tokenizer_trie.id2str(token_id)
            print(token, end="", flush=True)
            if token_id == tokenizer_trie.eos_token_id:
                print(f"\nFinished generating {i+1} tokens.")
                break
        print("\n")

        return overheads, masks, token_ids

    @pytest.mark.parametrize("test_name, schema", gen_all_json_schemas())
    def test_get_tokens_mask_extensive(
        self,
        tokenizer_trie: TokenizerTrie,
        prefix_cache: dict,
        test_name: str,
        schema: str,
    ):
        if test_name in EXCLUDED_PERFORMANCE_TESTS:
            pytest.skip(f"Skipping performance test {test_name}")
        print(f"[{test_name}] Testing get_tokens_mask with schema\n`{schema}`")
        overheads, *_ = self._run_random_sampling(
            schema=schema,
            tokenizer_trie=tokenizer_trie,
            prefix_cache=prefix_cache,
            verbose=False)

        print_stats(np.array(overheads) * 1000, name="ITL Overhead (ms)")

        # Performance thresholds.
        # Might need adjustment as we improve the latency.
        assert np.max(overheads) < 2, (
            "Max overhead should be less than 2 sec.")
        assert np.median(overheads) < 0.035, (
            "Median overhead should be less than <35 ms.")

    @pytest.mark.parametrize("test_name, schema", gen_all_json_schemas())
    def test_prefix_cache(
        self,
        tokenizer_trie: TokenizerTrie,
        prefix_cache: dict,
        test_name: str,
        schema: str,
    ):
        """Tests the masks array at each iteration with cache is the same as
        without cache."""
        print(f"[{test_name}] with schema\n`{schema}`")
        _, masks_w_cache, token_ids_w_cache, = self._run_random_sampling(
            schema=schema,
            tokenizer_trie=tokenizer_trie,
            prefix_cache=prefix_cache,
            verbose=False)

        _, masks_wo_cache, token_ids_wo_cache = self._run_random_sampling(
            schema=schema,
            tokenizer_trie=tokenizer_trie,
            prefix_cache=None,
            verbose=False)

        num_allowed_tokens_wo_cache = [mask.sum() for mask in masks_wo_cache]
        num_allowed_tokens_w_cache = [mask.sum() for mask in masks_w_cache]

        assert len(masks_w_cache) == len(masks_wo_cache)
        assert token_ids_w_cache == token_ids_wo_cache
        assert num_allowed_tokens_w_cache == num_allowed_tokens_wo_cache
        for mask_w_cache, mask_wo_cache in zip(masks_w_cache, masks_wo_cache):
            assert np.array_equal(mask_w_cache, mask_wo_cache)


class TestGrammarEnforcerCoreComponents:
    """Tests the core components of the GrammarEnforcer.
    """

    def test_constructor(self, tokenizer_trie: TokenizerTrie):
        grammar = Grammar.from_str("root ::= \"a\"")
        ge = GrammarEnforcer(grammar=grammar, tokenizer_trie=tokenizer_trie)
        assert ge.grammar == grammar
        assert ge.tokenizer_trie == tokenizer_trie

        with pytest.raises(ValueError):
            ge.get_tokens_mask()

        with pytest.raises(ValueError):
            ge.accept_token(0)

    def test_from_hf_tokenizer(self, hf_tokenizer: "PreTrainedTokenizerBase"):
        grammar = Grammar.from_str("root ::= \"a\"")
        ge = GrammarEnforcer.from_hf_tokenizer(grammar, hf_tokenizer)
        assert ge.tokenizer_trie == TokenizerTrie(hf_tokenizer)

    def test_accept_char_basic_charset(self, tokenizer_trie: TokenizerTrie):
        grammar = Grammar.from_str("root ::= [a-z0-9A-Z?]*")
        ge = GrammarEnforcer(grammar=grammar, tokenizer_trie=tokenizer_trie)
        ge.init()

        assert ge.accept_char("a")
        ge.init()
        assert ge.accept_char("1")
        ge.init()
        assert ge.accept_char("?")
        ge.init()
        assert ge.accept_char("A")
        assert ge.accept_char("B")

    def test_accept_char_basic_star(self, tokenizer_trie: TokenizerTrie):

        grammar = Grammar.from_str("root ::= \"a\"*\"b\"")
        ge = GrammarEnforcer(grammar=grammar, tokenizer_trie=tokenizer_trie)
        ge.init()
        assert ge.accept_char("a")
        ge.init()
        assert ge.accept_char("b")
        ge.init()
        assert not ge.accept_char("c")
        ge.init()
        assert ge.accept_char("a")
        assert ge.accept_char("a")
        assert ge.accept_char("b")

    def test_accept_char_basic_plus(self, tokenizer_trie: TokenizerTrie):
        grammar = Grammar.from_str("root ::= \"a\"+\"b\"")
        ge = GrammarEnforcer(grammar=grammar, tokenizer_trie=tokenizer_trie)
        ge.init()
        assert ge.accept_char("a")
        ge.init()
        assert not ge.accept_char("b")
        ge.init()
        assert not ge.accept_char("c")
        ge.init()
        assert ge.accept_char("a")
        assert ge.accept_char("a")
        assert ge.accept_char("b")

    def test_get_tokens_mask_plus(self, hf_tokenizer, tokenizer_trie):
        grammar = Grammar.from_str("root ::= \"a\"+\"b\"")
        ge = GrammarEnforcer(grammar=grammar, tokenizer_trie=tokenizer_trie)
        ge.init()
        mask = ge.get_tokens_mask()
        masked_token_ids = list(np.where(mask)[0])
        masked_tokens = set(
            hf_tokenizer.convert_ids_to_tokens(masked_token_ids))
        assert masked_tokens == {"ab", "aaaa", "aa", "a"}

    def test_get_tokens_mask_star(self, hf_tokenizer, tokenizer_trie):
        grammar = Grammar.from_str("root ::= \"a\"*\"b\"")
        ge = GrammarEnforcer(grammar=grammar, tokenizer_trie=tokenizer_trie)
        ge.init()
        mask = ge.get_tokens_mask()
        masked_token_ids = list(np.where(mask)[0])
        masked_tokens = set(
            hf_tokenizer.convert_ids_to_tokens(masked_token_ids))
        assert masked_tokens == {"ab", "aaaa", "aa", "a", "b"}

    @pytest.mark.parametrize("root_name, num_expected_tokens", [
        ("string", 36),
        ("string_1", 2156),
        ("string_2", 31606),
        ("string_1_1", 23),
    ])
    def test_get_tokens_mask_string(self, tokenizer_trie, root_name,
                                    num_expected_tokens, prefix_cache):
        """Tests that the number of tokens for some special rules are as
        expected. It also checks that using caches vs. not using cache results
        in the same allowed tokens."""
        grammar = Grammar.from_str(
            f"root ::= {root_name} \n string ::= {STRING} \n hexdigit ::= {HEX_DIGIT} \nspace ::= {SPACE_RULE}\nspace_e ::= {SPACE_END}"  # noqa
        )

        ge_with_cache = GrammarEnforcer(grammar=grammar,
                                        tokenizer_trie=tokenizer_trie,
                                        global_stack_prefix_cache=prefix_cache)
        ge_with_cache.init()
        mask_with_cache = ge_with_cache.get_tokens_mask()
        allowed_tokens_with_cache = set(
            tokenizer_trie.id2str(id) for id in np.where(mask_with_cache)[0])

        ge_without_cache = GrammarEnforcer(
            grammar=grammar,
            tokenizer_trie=tokenizer_trie,
        )
        ge_without_cache.init()
        mask_without_cache = ge_without_cache.get_tokens_mask()
        allowed_tokens_without_cache = set(
            tokenizer_trie.id2str(id)
            for id in np.where(mask_without_cache)[0])

        assert len(allowed_tokens_without_cache) == num_expected_tokens
        assert allowed_tokens_with_cache == allowed_tokens_without_cache
