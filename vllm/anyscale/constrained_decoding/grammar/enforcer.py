import os
from collections import deque
from typing import FrozenSet, Hashable, List, Optional, Set, Tuple, Union, cast

import numpy as np
from cachetools import LRUCache
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .cache import PrefixCache
from .element import Element
from .parser import Grammar
from .stack import Stack
from .trie import TokenizerTrie, Trie

# 2 ** 10
MAX_GRAMMAR_ENFORCER_LOCAL_CACHE_SIZE = int(
    os.environ.get("MAX_GRAMMAR_ENFORCER_LOCAL_CACHE_SIZE", 1 << 10))
# 2 ** 16
MAX_GRAMMAR_ENFORCER_ADVANCE_CACHE_SIZE = int(
    os.environ.get("MAX_GRAMMAR_ENFORCER_ADVANCE_CACHE_SIZE", 1 << 16))

LocalCacheKeyType = FrozenSet[Stack]
AdvanceStackCacheKeyType = Tuple[Tuple[Element, ...], Stack, bool]


class GrammarEnforcer:
    """Enforces the grammar on the input tokens.

    Args:
        grammar: The grammar object that captures the rules that we want to
            enforce.
        tokenizer_trie: To avoid creating the tokenizer trie each time we
            create a grammar, you can create it outside and pass it in, so it
            gets reused.
        global_stack_prefix_cache: A dictionary that maps stack prefixes to
            token masks. This is useful for handling common special cases that
            require a big trie walk but you want to shortcut it, by matching a
            stack-prefix that you know the token masks for. This can speed up
            json mode decoding significantly (for cases where the rule is to
            decode a string)
    """

    def __init__(
        self,
        grammar: Grammar,
        tokenizer_trie: TokenizerTrie,
        global_stack_prefix_cache: Optional["PrefixCache"] = None,
    ) -> None:
        self.grammar = grammar

        # stacks holds the state of the enforcer.
        # Each item in the stack counts for one possible stack of rule elements
        # that can be matched. It implemented as a Set to avoid duplicates.
        self.stacks: Set[Stack] = set()
        self.tokenizer_trie = tokenizer_trie

        # Global cache passed in externally:
        # Should be a mapping from stack prefixes to token masks.
        # If a given stack prefix is found in the stack at any time, we return
        # the token mask corresponding to that prefix without doing a trie walk.
        self.global_stack_prefix_cache = (global_stack_prefix_cache
                                          or PrefixCache())
        # Local cache from stacks to token_mask:
        # If during multi-step generation, we find ourselves in the same state
        # as given by the stacks we can return the token mask directly. There
        # are many schemas that for many steps, the state remains idempotent,
        # meaning that the stack remains the same even after accepting many
        # tokens. In these cases you won't need to calculate anything.
        # In this cache, keys are tuples of stacks and values are token masks.
        self._local_stack_cache: LRUCache[LocalCacheKeyType, np.ndarray] = (
            LRUCache(MAX_GRAMMAR_ENFORCER_LOCAL_CACHE_SIZE))
        # Cache for advance_stack:
        # The keys in this cache are tuple of elements in the rule and the
        # input_stack. If the schema has this property that many rules
        # reference essentially the same rule (for example there are a lot of
        # rules that reference string), then this cache can be useful. In these
        # scenarios, the rule that we are adding will be, for example,
        # `string`, and if the input_stack is also the same, we can return the
        # updated stack values without having to go through the recursion.
        self._advance_stack_cache: LRUCache[Hashable, Set[Stack]] = (
            LRUCache(MAX_GRAMMAR_ENFORCER_ADVANCE_CACHE_SIZE))

        self._initialized = False

    @classmethod
    def from_hf_tokenizer(
        cls,
        grammar: Grammar,
        tokenizer: Union[PreTrainedTokenizerBase, str],
        global_stack_prefix_cache: Optional["PrefixCache"] = None,
    ) -> "GrammarEnforcer":
        """Creates a GrammarEnforcer from a grammar and an hf tokenizer object.
        """

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

        tokenizer_trie = TokenizerTrie(tokenizer)
        return cls(grammar, tokenizer_trie, global_stack_prefix_cache)

    def init(self, stacks: Optional[List[Stack]] = None) -> None:
        """Initializes the state of the stack set.

        You can either initialize the state of the enforcer via an external
        stack (used for testing) or via the grammar.

        Args:
            stacks: The initial state of the stack set. If None, the
                initial state is set to the start rule of the grammar.
        """
        start_rule_name = self.grammar.start_rule_name
        start_rule = self.grammar.rules[start_rule_name]

        if stacks is not None:
            self.stacks = set(stacks)
        else:
            # When initialized we don't resolve the reference rules to get
            # cache hit with the common prefixes
            self.stacks = self.advance_stack(rule=start_rule,
                                             rule_name=start_rule_name,
                                             resolve_references=False)
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def _raise_error_if_not_initialized(self, method_name: str) -> None:
        if not self.is_initialized():
            raise ValueError(
                "EnforcerState needs to be initialized before calling "
                f"this .{method_name}. Hint: call `init()` method.")

    def _accept_char(self, char: str, update_stack: bool = False) -> bool:
        """Checks if a character can be matched to the current state.

        This method changes the internal state of the enforcer if the character
        is matched and `update_stack` is true. If `update_stack` is False, it
        will just check if the character can be matched.

        Args:
            char: The character to be matched.
            update_stack: If True, the internal state of the enforcer will be
                updated if the character is matched, otherwise it will just not
                change the state of the enforcer.
        Returns:
            True if the character is matched. If False is returned,
            the state is not updated.
        """
        is_matched = False
        new_stacks: Set[Stack] = set()

        queue = deque(self.stacks)
        while queue:
            stack = queue.popleft()
            top_element = stack.top()
            if top_element is not None:
                if top_element.is_rule_ref():
                    stacks = self.advance_stack(stack=stack)
                    queue.extend(stacks)
                elif top_element.is_match(ord(char)):
                    is_matched = True
                    if not update_stack:
                        return is_matched
                    new_stack = cast(Stack, stack[1:])
                    # DO NOT expand the references within this stack to
                    # maximize cache hits for the next step of get_token_masks
                    new_stack_set = self.advance_stack(
                        stack=new_stack, resolve_references=False)
                    new_stacks.update(new_stack_set)

        if update_stack and is_matched:
            self.stacks = new_stacks

        return is_matched

    def accept_char(self, char: str) -> bool:
        """Checks if a char can be matched to the current state.

        NOTE: This method updates the state only if the character is accepted.
        """
        self._raise_error_if_not_initialized("accept_char")
        return self._accept_char(char, update_stack=True)

    def accept_token(self, token_id: int) -> bool:
        """Checks if a token can be matched to the current state.

        NOTE: This method changes the internal state of the enforcer, whether
        the token is fully accepted or partially accepted.

        NOTE: In LLM tokenizers, a given token may be very similar to the token
        that should be matched, but still it does not get matched. For example,
        the token `▁{"` has a space in front, but the grammar rule dictates
        `{"` as the start of the string. In other words, you cannot always
        take a response that looks like it is a match, tokenize it and expect
        it to be matched by the grammar. In that case you should match only by
        characters as going through tokenizer may give you incorrect answers.
        """
        self._raise_error_if_not_initialized("accept_token")

        # Always accept EOS token regardless of the current state.
        # If we do not do this, then in case of ignore_eos=True,
        # the enforcer will not accept the EOS token and will cause
        # failure in the logit processor
        if token_id == self.tokenizer_trie.eos_token_id:
            return True

        for c in self.tokenizer_trie.id2str(token_id):
            if not self.accept_char(c):
                return False
        return True

    def _advance_stack(self,
                       *,
                       rule: List[Element],
                       stack: Stack,
                       rule_name: str = "",
                       resolve_references: bool = True) -> Set[Stack]:
        """Returns a list of new options based on the given rule and stack.

        This method is the most critical function of enforcer that keeps the
        internal state up-to-date. It is also performance critical.

        This method has two modes:
        1. If rule is specified, it will add the rule to the top of the stack
        while maintaining all the options when an ALT element is encountered.
        2. If rule is not specified, it will resolve the potential rule
        references on top of the stack and recursively maintains possible
        alternatives. The output will be a list of all possible stacks after
        advancing the stack.

        Args:
            rule: The rule to be added to the top of the stack. This is a list
                of elements that need to be iterated over and added to the top
                of the stack(s).
            stack: The initial stack to be advanced.
            rule_name: What rule name is being added to the stack. This should
                non-empty string when rule is not empty.
            resolve_references: If True, the references in the rule will be
                resolved to their primitive counterparts. If False, the
                references will not be resolved.

        Returns:
            A set of new stacks after advancing the given stack.
        """

        updated_stacks = set()
        if rule_name:
            assert len(rule) > 0
            assert rule[-1].is_end()
            new_stacks: Set[Stack] = set()

            init_stack_size = stack.size()

            for element in reversed(rule):
                if element.is_end():
                    continue
                elif element.is_alt():
                    # If element is alt (|) we need to fork the stack.
                    # Forking is done by saving a snapshot of the current stack
                    # and adding it to new_stacks set and then continuing the
                    # process with a stack that is initialized with the bottom
                    # of the original input stack.
                    if not stack.is_empty() and stack not in new_stacks:
                        new_stacks.add(stack)
                    # The number of items we have added to the stack
                    # compared to the original input stack.
                    start_index = stack.size() - init_stack_size
                    stack = cast(Stack, stack[start_index:])
                else:
                    stack.push(element)

            if not stack.is_empty() and stack not in new_stacks:
                new_stacks.add(stack)

            for new_stack in new_stacks:
                updated_stacks.update(self.advance_stack(stack=new_stack))
        else:
            top_element = stack.top()
            if top_element is None:
                pass
            elif top_element.is_rule_ref() and resolve_references:
                assert top_element.rule_name is not None
                ref_rule_name = top_element.rule_name
                referenced_rule = self.grammar.rules[ref_rule_name]

                # If we hit a reference rule, we need to resolve it and add it
                # on top of the stack below the top element.
                new_stack = cast(Stack, stack[1:])
                new_updated_stacks = self.advance_stack(
                    rule=referenced_rule,
                    stack=new_stack,
                    rule_name=ref_rule_name,
                    resolve_references=resolve_references)
                updated_stacks.update(new_updated_stacks)
            else:
                updated_stacks.add(stack)

        return updated_stacks

    def advance_stack(self,
                      *,
                      rule: Optional[List[Element]] = None,
                      stack: Optional[Stack] = None,
                      rule_name: str = "",
                      resolve_references: bool = True) -> Set[Stack]:
        """Implements caching at advance_stack level."""
        rule = rule or []
        if stack is None:
            stack = Stack()

        cache_key = (tuple(rule), stack, rule_name, resolve_references)
        if cache_key in self._advance_stack_cache:
            return self._advance_stack_cache[cache_key]

        updated_stacks = self._advance_stack(
            rule=rule,
            stack=stack,
            rule_name=rule_name,
            resolve_references=resolve_references)
        self._advance_stack_cache[cache_key] = updated_stacks
        return updated_stacks

    def get_tokens_mask(self) -> np.ndarray:
        """Returns a mask over allowed tokens, given the current state.

        Output is a numpy array of shape (num_vocab,) where true indicates
        allowed tokens.

        NOTE: This method does not update the state. To update the state, use
        `accept_token`.

        Returns:
            A numpy array of shape (num_vocab,) where true indicates allowed
            tokens.
        """
        self._raise_error_if_not_initialized("get_tokens_mask")

        cache_key = frozenset(self.stacks)
        if cache_key in self._local_stack_cache:
            return self._local_stack_cache[cache_key]

        # Using numpy is more efficient than lists or torch tensors.
        token_mask = np.zeros(self.tokenizer_trie.vocab_size, dtype=np.bool_)

        for stack in self.stacks:

            self.walk_trie(self.tokenizer_trie,
                           stack,
                           token_mask,
                           cache=self.global_stack_prefix_cache)

        if not np.any(token_mask):
            # If there is no option, we should unmask eos token
            token_mask[self.tokenizer_trie.eos_token_id] = True

        self._local_stack_cache[cache_key] = token_mask
        return token_mask

    def walk_trie(
        self,
        trie: Trie,
        stack: Stack,
        mask: np.ndarray,
        cache: Optional["PrefixCache"] = None,
    ) -> List[Trie]:
        """Iterative function to walk the trie and update the mask.

        This method, given a stack of rule elements, walks the tokenizer trie,
        character by character, and checks if the character can be accepted by
        the top element of the stack. If so it updates the stack and checks the
        next character, and so on. During this iterative process when the
        traversal encounters a token_id where all the corresponding characters
        are accepted by the stack, it updates the mask array to accept that
        token_id.

        Args:
            trie: The trie to be walked.
            stack: The stack of rule elements. This is the initial state at the
                root of the trie.
            mask: The mask array that is updated to accept the token_ids that
                are valid for the given stack. This is updated in-place.
            cache: Optional external cache to store the prefix of stacks and
                the tries.
        Returns:
            A list of prefix trees that are not explored due to reaching the
            end of the stack. This list is used as the values of the cache so
            that when we get a stack prefix match we can resume the state of
            trie walk from the latest point.

            Also the mask is updated in-place.
        """
        cache = cache or PrefixCache()
        trie_queue = deque([(trie, stack)])

        unexplored_tries = []
        while trie_queue:
            current_trie, current_stack = trie_queue.popleft()
            if current_trie.is_valid():
                mask[current_trie.value] = True

            cache_key = (current_stack, current_trie)
            if cache_key in cache:
                cached_mask, cached_tries = cache[cache_key]
                if cached_mask is not None:
                    mask |= cached_mask

                    stack_prefix = cache.get_matched_stack_prefix(stack)
                    remaining_stack = cast(Stack, stack[stack_prefix.size():])
                    if cached_tries:
                        trie_queue = deque([(trie, remaining_stack)
                                            for trie in cached_tries])
                    continue

            top_element = current_stack.top()
            if top_element is not None:

                if top_element.is_rule_ref():
                    stacks = self.advance_stack(stack=current_stack)
                    for nstack in stacks:
                        trie_queue.append((current_trie, nstack))
                    continue
                elif top_element.is_matchable():
                    for char, sub_trie in current_trie.items():
                        if top_element.is_match(ord(char)):
                            new_stack = cast(Stack, current_stack[1:])

                            if new_stack.is_empty():
                                # It means that we have reached the end of the
                                # stack
                                unexplored_tries.append(sub_trie)

                            trie_queue.append((sub_trie, new_stack))

        return unexplored_tries
