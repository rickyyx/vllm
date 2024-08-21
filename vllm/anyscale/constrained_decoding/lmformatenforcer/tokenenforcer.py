# ruff: noqa

import logging
import os
from dataclasses import dataclass, field
from typing import (Callable, Dict, FrozenSet, Hashable, List, Optional, Set,
                    Tuple)

from vllm.transformers_utils.detokenizer import detokenize_incrementally
from vllm.utils import LRUCache

from .characterlevelparser import (CharacterLevelParser,
                                   CharacterLevelParserConfig, ForceStopParser)
from .exceptions import LMFormatEnforcerException
from .tokenizerprefixtree import TokenizerPrefixTree, TokenizerPrefixTreeNode

logger = logging.getLogger(__name__)

# We can set CACHE_SIZE relatively high as we will not store past states
# above CACHE_TOKENS_LEN_THRESHOLD (aside from the last one).
# TODO: Use memory size/number of tokens as a way to determine
# cache size instead of number of entries.
CACHE_SIZE = int(
    os.getenv(
        "ANYSCALE_VLLM_LM_FORMAT_TOKEN_ENFORCER_PREFIX_STATES_CACHE_SIZE",
        str(4096)))
CACHE_TOKENS_LEN_THRESHOLD = int(
    os.getenv(
        "ANYSCALE_VLLM_LM_FORMAT_TOKEN_ENFORCER_PREFIX_STATES_CACHE_TOKENS_LEN_THRESHOLD",
        str(128)))


class TokenEnforcerTokenizerData:
    """TokenEnforcerTokenizerData contains all of the preprocessing for preparing the TokenEnforcer to work with a 
    specific tokenizer. It does some calculations, so it is recommended to reuse it for multiple TokenEnforcers"""

    def __init__(self, regular_tokens: List[Tuple[int, str, bool]],
                 decoder: Callable[[List[int]], str], eos_token_id: int):
        """
        Create the tokenizer data that the TokenEnforcer needs. This can be reused for multiple TokenEnforcers if they work with the same tokenizer.
        :param regular_tokens: A list of tuples (token_id, token_string, is_new_word_token) for all the regular (not special) tokens in the tokenizer vocabulary.
        Note that token_string is expected to include leading / trailing whitespaces if relevant.
        :param decoder: A function that decodes a list of token ids into a string.
        :param eos_token_id: The token id of the end-of-string token.
        """
        self.regular_tokens = regular_tokens
        self.tokenizer_tree = TokenizerPrefixTree(regular_tokens)
        self.decoder = decoder
        self.eos_token_id = eos_token_id
        self.tokenizer_alphabet = "".join(
            token_str
            for token_str in self.tokenizer_tree.root.children.keys()
            if len(token_str) == 1)


class TokenEnforcer:
    """TokenEnforcer provides a token filtering mechanism, given a CharacterLevelParser and some information about the tokenizer.
    It is the main entry point for extending lm-format-enforcer to new inference libraries. See __init__() and get_allowed_tokens()"""

    @dataclass
    class OutputTensorState:
        parser: CharacterLevelParser
        allowed_tokens: Optional[FrozenSet[int]] = None
        current_word_tokens: List[int] = field(default_factory=list)
        current_word_token_strs: Optional[List[str]] = None
        prefix_offset: int = 0
        read_offset: int = 0

    def __init__(self, tokenizer_data: TokenEnforcerTokenizerData,
                 parser: CharacterLevelParser):
        """
        Create a new TokenEnforcer.
        :param tokenizer_data: Per tokenizer data that the token enforcer needs in order to operate.
        :param parser: A CharacterLevelParser that defines the allowed strings.
        """
        self.prefix_states: Dict[
            Tuple, TokenEnforcer.OutputTensorState] = LRUCache(CACHE_SIZE)

        self.root_parser = parser
        self.tokenizer_tree = tokenizer_data.tokenizer_tree
        self.decoder = tokenizer_data.decoder
        self.eos_token_id = tokenizer_data.eos_token_id
        self.regular_tokens = tokenizer_data.regular_tokens
        self.allowed_token_cache: Dict[Hashable, List[int]] = {}

        config = CharacterLevelParserConfig(
            alphabet=tokenizer_data.tokenizer_alphabet)
        parser.config = config

    def get_allowed_tokens(
            self,
            token_sequence: List[int],
            candidate_tokens: Optional[List[int]] = None) -> FrozenSet[int]:
        """
        Get a list of allowed tokens, given a list of tokens that were already generated.
        :param token_sequence: The tokens that were already generated, and the next token will be generated for.
        :param candidate_tokens: A list of candidate tokens that are to be used for constraining the
            search on allowed next tokens. This is to speed up the search. The more candidate tokens there are,
            the greater the response quality, however the slower the response time. If candidate tokens is
            None or empty, then all tokens are considered candidates.
        :return: A list of token ids that are allowed to be selected next.
        """
        # In order to elegantly support beam search and batching, we don't store per-batch information.
        # Instead, we store a hash of all the states (unique token tensors) we encountered so far.
        # When we encounter a new unique token tensor, we find the token tensor that led to it, and continue from there.
        sent_tuple = tuple(token_sequence)
        prev_step_tuple = sent_tuple[:-1]

        if sent_tuple in self.prefix_states:
            # We already calculated for this node, return cached list
            return self.prefix_states[sent_tuple].allowed_tokens
        elif prev_step_tuple not in self.prefix_states:
            # We have not encountered the tensor up to the before-last entry. This means that this is the first call - the instruction / prompt tensor.
            # Initialize the root node
            state = TokenEnforcer.OutputTensorState(parser=self.root_parser)
            self.prefix_states[sent_tuple] = state
            self._compute_allowed_tokens(sent_tuple, state, candidate_tokens)
            return state.allowed_tokens
        else:
            # Find the state that led to this node. We explicitly don't use the concept of "timestep" because of beam search
            prev_step_state = self.prefix_states[prev_step_tuple]
            new_state = self._apply_new_characters(prev_step_state,
                                                   token_sequence)
            if len(prev_step_tuple) > CACHE_TOKENS_LEN_THRESHOLD:
                self.prefix_states.pop(prev_step_tuple, None)
            self.prefix_states[sent_tuple] = new_state
            self._compute_allowed_tokens(sent_tuple, new_state,
                                         candidate_tokens)
            return new_state.allowed_tokens

    def _compute_allowed_tokens(self,
                                state_tokens: Tuple,
                                state: 'TokenEnforcer.OutputTensorState',
                                candidate_tokens: Optional[List[int]] = None):
        try:
            allowed_tokens: Set[int] = set()
            cache_key = state.parser.cache_key()
            if cache_key is not None and cache_key in self.allowed_token_cache:
                logger.info(f"Using cached allowed tokens for {cache_key}")
                state.allowed_tokens = self.allowed_token_cache[cache_key]
                return
            shortcut_key = state.parser.shortcut_key()
            if candidate_tokens:
                candidate_token_strs = {
                    self.tokenizer_tree.tokens_to_strs.get(token, "")
                    for token in candidate_tokens
                }

                self._collect_allowed_tokens(state.parser,
                                             self.tokenizer_tree.root,
                                             allowed_tokens, shortcut_key,
                                             candidate_token_strs, 0)
            if not allowed_tokens:
                if candidate_tokens:
                    logger.info("Did not find and matching candidate tokens")
                self._collect_allowed_tokens(state.parser,
                                             self.tokenizer_tree.root,
                                             allowed_tokens, shortcut_key)
            if state.parser.can_end():
                allowed_tokens.add(self.eos_token_id)
            if not allowed_tokens:
                raise ValueError(
                    f"Parser reached state with no allowed tokens")
            # root_state = next(state for state in self.prefix_states.values() if state.parser == self.root_parser)
            # print(f"Allowing {len(allowed_tokens)} tokens after {state.str_so_far[len(root_state.str_so_far):]}")
            allowed_tokens = frozenset(allowed_tokens)
            state.allowed_tokens = allowed_tokens
            if cache_key is not None:
                self.allowed_token_cache[cache_key] = allowed_tokens
        except LMFormatEnforcerException:
            # Getting an LMFormatEnforcerException means that we know what the user did wrong,
            # and we can give a nice error message for them to fix.
            raise
        except Exception:
            # Other exceptions are potential bugs and should be reported
            logging.basicConfig(
                level=logging.ERROR)  # Initialize if no loggers
            prefix = self.decoder(list(state_tokens))
            logger.exception(
                f"Unknown LMFormatEnforcer Problem. Prefix: '{prefix}'\n"
                "Terminating the parser. Please open an issue at \n"
                "https://github.com/noamgat/lm-format-enforcer/issues with the prefix and "
                "CharacterLevelParser parameters")
            state.allowed_tokens = frozenset((self.eos_token_id, ))

    def _collect_allowed_tokens(self,
                                parser: CharacterLevelParser,
                                tree_node: TokenizerPrefixTreeNode,
                                allowed_tokens: Set[int],
                                shortcut_key: Optional[Hashable],
                                candidate_token_strs: Set[str] = None,
                                current_char_idx: int = 0):
        allowed_tokens.update(tree_node.tokens)
        allowed_characters = parser.get_allowed_characters()
        relevant_characters = tree_node.children.keys()
        # This next line is the heart of the traversal algorithm. We only explore paths that are shared by both the parser and the tokenizer.
        characters_to_explore = set(relevant_characters).intersection(
            allowed_characters)

        # Performance optimization: If we are in JSON freetext, all of the tokens that don't contain quote, or end with quote, are legal, so we take
        # their cached list. If the quote character is allowed, we only need to dynamically explore the cases where the string starts with a quote.
        # This breaks the elegance of the API, but otherwise it is a huge performance hit.
        if isinstance(shortcut_key, tuple):
            if shortcut_key[0] == 'json_freetext':
                assert len(shortcut_key) == 4
                _, cur_len, min_len, max_len = shortcut_key
                cache = self.tokenizer_tree.json_freetext_tokens

                min_remaining = min(
                    cache.max_token_len,
                    max(0, min_len -
                        cur_len))  # no " allowed before this many chars
                max_allowed_len = min(
                    cache.max_token_len,
                    max_len - cur_len)  # max new characters allowed (before ")

                allowed_tokens.update(
                    cache.lookup_allowed_tokens(min_remaining,
                                                max_allowed_len))
                characters_to_explore = characters_to_explore.intersection(
                    ['"'])
            elif shortcut_key[0] == 'json_backslash':
                cache = self.tokenizer_tree.backslash_tokens
                allowed_tokens.update(cache.lookup_allowed_tokens())
                characters_to_explore = characters_to_explore.intersection(
                    ['u', '"'])

        # performance optimization: Accept a set of top likely candidate token
        # strings, and only explore characters that are in the candidate token
        # strings. Add the current_char_idx character of each string to the
        # set of characters to explore.
        if candidate_token_strs:
            to_remove = set()
            for candidate_str in candidate_token_strs:
                if current_char_idx > len(candidate_str) - 1:
                    to_remove.add(candidate_str)
            candidate_token_strs = candidate_token_strs - to_remove
            if not candidate_token_strs:
                return
            current_candidate_chars = {
                candidate_str[current_char_idx]
                for candidate_str in candidate_token_strs
            }
            characters_to_explore = characters_to_explore.intersection(
                current_candidate_chars)
            current_char_idx += 1

        for character in characters_to_explore:
            next_parser = parser.add_character(character)
            next_tree_node = tree_node.children[character]
            self._collect_allowed_tokens(next_parser, next_tree_node,
                                         allowed_tokens, None,
                                         candidate_token_strs,
                                         current_char_idx)

    def _apply_new_characters(self, state: 'TokenEnforcer.OutputTensorState',
                              token_sequence: List[int]):
        new_state = TokenEnforcer.OutputTensorState(parser=state.parser)
        new_token = token_sequence[-1]
        if new_token in self.tokenizer_tree.new_word_tokens:
            new_state.current_word_tokens = [new_token]
            new_characters = self.tokenizer_tree.tokens_to_strs[new_token]
        else:
            new_state.current_word_tokens = state.current_word_tokens + [
                new_token
            ]
            (new_tokens, new_characters, prefix_offset,
             read_offset) = detokenize_incrementally(
                 self.decoder,
                 new_state.current_word_tokens,
                 state.current_word_token_strs,
                 state.prefix_offset,
                 state.read_offset,
             )
            if state.current_word_token_strs:
                new_tokens = state.current_word_token_strs + new_tokens
            new_state.current_word_token_strs = new_tokens
            new_state.prefix_offset = prefix_offset
            new_state.read_offset = read_offset
        for character in new_characters:
            if character in new_state.parser.get_allowed_characters():
                new_state.parser = new_state.parser.add_character(character)
            else:
                # This can happen in beam / batch scenarios, when some of the batches finished but others are continuing.
                logging.debug(
                    f"Received an invalid character '{character}', switching to ForceStopParser"
                )
                new_state.parser = ForceStopParser()
        return new_state
