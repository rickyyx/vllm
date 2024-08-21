from typing import TYPE_CHECKING

import numpy as np

from .cache import PrefixCache
from .element import Element, ElementType
from .enforcer import GrammarEnforcer
from .parser import Grammar
from .stack import Stack

if TYPE_CHECKING:
    from .trie import TokenizerTrie

# These special cases are derived by looking at some json schemas that took
# longer than average and observing the most common stack prefix for
# decoding steps that took the most time. For example "type": "string" usually
# results in some of the following stack prefixes that can be cached.
SPECIAL_CASES = (
    Stack(
        reversed([
            Element(ElementType.RULE_REF, rule_name="string_1_1"),
            Element(ElementType.RULE_REF, rule_name="string_2"),
            Element(ElementType.CHAR, code_point=ord("\"")),
        ])),
    Stack(
        reversed([
            Element(ElementType.RULE_REF, rule_name="string_2"),
            Element(ElementType.CHAR, code_point=ord("\"")),
        ])),
    # Add more special cases here
)


def compute_json_mode_cache(tokenizer_trie: "TokenizerTrie") -> PrefixCache:
    """Computes the global cache for JSON mode for a given tokenizer trie.

    There exist grammars where the search space will end up being large and you
    have no choice but to traverse the entirety of the trie to find that most
    of the tokens that are allowed. However, with json mode, we control the
    way the grammar is written and because of that we can find the most common
    rule prefixes on top of stacks that are likely to be encountered that can
    result in slow-downs during decoding. We can cache the result of such cases
    and for every iteration check if such prefix of elements is found in the
    top of the current stack. If so, we can skip the trie traversal and return
    the cached results.

    `SPECIAL_CASES` is a list of prefixes that commonly happen during parsing
    generation of string types within json. We can only do this in json (and
    not the generic grammar) because the construction of the key for caching is
    based on arbitrary rule reference names. In json, these rules are fixed, as
    long as the json schema to grammar conversion is deterministic. So, we can
    take advantge of that determinism to cache the results of these stacks.

    NOTE: That if the rules with which we express json schemas as
    GBNF grammars change, we need to update the SPECIAL_CASES list.

    Args:
        tokenizer_trie: The trie that contains the tokens that are allowed in
        the grammar.
    Returns:
        A dictionary where the keys are the tuples of stack, trie. Stack are
        prefixes that are in `SPECIAL_CASES` and for generality we also use
        tokenizer trie as a key here, and the values are the corresponding mask
        arrays across the tokenizer vocabulary and the unexplored sub-tries
        where the trie walk can start from if there is a prefix match.
    """
    cache = PrefixCache()

    # We initialize a grammar with a generic json schema and then use that to
    # construct the cache dict
    grammar = Grammar.from_json_schema({})
    ge = GrammarEnforcer(grammar, tokenizer_trie=tokenizer_trie)
    for stack in SPECIAL_CASES:
        # Make sure the stack is frozen so that it can be used keys in a dict.
        stack.freeze()
        # We should initialize the state of the enforcer in each round
        ge.init()
        mask = np.zeros(tokenizer_trie.vocab_size, dtype=np.bool_)

        unexplored_tries = ge.walk_trie(ge.tokenizer_trie, stack, mask)
        cache[stack, ge.tokenizer_trie] = (mask, unexplored_tries)
    return cache
