import gc
import json
import logging
import os
import time
from typing import FrozenSet, List, Set, Tuple, cast

import msgspec
import numpy as np
import psutil
import torch
from anyguide import (Grammar, GrammarEnforcer, TokenizerTrie,
                      compute_json_mode_cache)
from cachetools import LRUCache
from transformers import PreTrainedTokenizerBase

from vllm.anyscale.constrained_decoding.fault_tolerance import FaultAwareDaemon
from vllm.anyscale.shm.msgspec_shm import SharedMsgspecBufferWithEvent
from vllm.transformers_utils.tokenizer import get_tokenizer

from .lmformatenforcer import JsonSchemaParser
from .lmformatenforcer.integrations.transformers import (
    CharacterLevelParser, TokenEnforcer, TokenEnforcerTokenizerData,
    _build_regular_tokens_list)

logger = logging.getLogger(__name__)

MAX_MEMORY_LIMIT = int(
    os.getenv("ANYSCALE_VLLM_JSON_LOGITS_PROCESSOR_MAX_MEMORY_BYTES",
              str(int(1.5e+10))))  # 15 GB
SCHEMA_CACHE_SIZE = int(
    os.getenv("ANYSCALE_VLLM_JSON_LOGITS_PROCESSOR_SCHEMA_CACHE_SIZE",
              str(256)))
TENSOR_CACHE_SIZE = int(
    os.getenv("ANYSCALE_VLLM_JSON_LOGITS_PROCESSOR_TENSOR_CACHE_SIZE",
              str(1024)))


def build_token_enforcer_tokenizer_data(
        tokenizer: PreTrainedTokenizerBase) -> TokenEnforcerTokenizerData:
    regular_tokens = _build_regular_tokens_list(tokenizer)

    # This is required to avoid many code changes inside lmformatenforcer
    class Decoder(tokenizer.__class__):

        def __call__(self, tokens: List[int]) -> str:
            decoded = tokenizer.decode(tokens)
            cleaned = decoded.rstrip("�")
            return cleaned

    tokenizer.__class__ = Decoder

    return TokenEnforcerTokenizerData(regular_tokens, tokenizer,
                                      tokenizer.eos_token_id)


def build_token_enforcer(
        tokenizer_data: TokenEnforcerTokenizerData,
        character_level_parser: CharacterLevelParser) -> TokenEnforcer:
    token_enforcer = TokenEnforcer(tokenizer_data, character_level_parser)
    return token_enforcer


class JSONLogitsProcessorInput(msgspec.Struct, array_like=True):
    input_list: List[Tuple[List[int], str]]


class JSONModeLogitsProcessor(FaultAwareDaemon):
    """A remote wrapper for a function for constraining model logits
    to a JSON schema.
    """

    def __init__(self,
                 rank: int,
                 tokenizer_name_or_path: str,
                 padded_vocab_size: int,
                 recreate_failed_actors: bool = False,
                 delay_between_actor_restarts_s: float = 0) -> None:
        super().__init__(recreate_failed_actors,
                         delay_between_actor_restarts_s)

        # Rank of the json mode logit processor. Testing-only.
        self._rank = rank
        self.tokenizer = get_tokenizer(tokenizer_name_or_path)
        self.padded_vocab_size = padded_vocab_size
        self.token_enforcer_cache = LRUCache(SCHEMA_CACHE_SIZE)
        self.tokenizer_data = build_token_enforcer_tokenizer_data(
            self.tokenizer)
        self.tensor_cache = LRUCache(TENSOR_CACHE_SIZE)
        self._process = psutil.Process()

    def add_schema(self, json_schema: str) -> TokenEnforcer:
        if json_schema not in self.token_enforcer_cache:
            schema_parser = JsonSchemaParser(json.loads(json_schema) or None)
            json_mode_logits_processor = (build_token_enforcer(
                self.tokenizer_data, schema_parser))
            self.token_enforcer_cache[json_schema] = json_mode_logits_processor
        return self.token_enforcer_cache[json_schema]

    def apply_mask(self, allowed_tokens: FrozenSet[int],
                   row: torch.Tensor) -> None:
        """Applies a mask derived from allowed_tokens to the row tensor
        in-place."""
        if allowed_tokens not in self.tensor_cache:
            idx = torch.tensor(list(allowed_tokens), dtype=torch.long)
            row = row.index_fill_(0, idx, True)
            # Cache the tensor so we can avoid having to recreate it
            # in the future (especially costly for mostly-full masks).
            self.tensor_cache[allowed_tokens] = row.clone()
            return

        mask = self.tensor_cache[allowed_tokens]
        row.copy_(mask)

    def call(self, input_list: List[Tuple[List[int], str]]) -> np.ndarray:
        allowed_token_tensor = torch.zeros(
            (len(input_list), self.padded_vocab_size), dtype=torch.bool)
        for i, data in enumerate(input_list):
            token_ids, schema = data
            token_ids = tuple(token_ids)
            token_enforcer = self.add_schema(schema)
            j = 0
            for j in range(len(token_ids), -1, -1):
                if token_ids[:j] in token_enforcer.prefix_states:
                    break
            for k in range(j, len(token_ids) + 1):
                allowed_tokens = token_enforcer.get_allowed_tokens(
                    token_ids[:k])
            self.apply_mask(allowed_tokens, allowed_token_tensor[i])
        return allowed_token_tensor.numpy()

    def daemon_setup(
            self,
            shared_memory_buffer_input: SharedMsgspecBufferWithEvent,
            shared_memory_buffer_output: SharedMsgspecBufferWithEvent,
            **kwargs  # pylint: disable=unused-argument
    ) -> None:
        """Setup the shared memory buffer for the JSON logits processor."""
        shared_memory_buffer_input.decoder = (
            msgspec.msgpack.Decoder(JSONLogitsProcessorInput))
        logger.info("JSON logits processor buffer id: "
                    "%d "
                    "%d", shared_memory_buffer_input.participant_id,
                    shared_memory_buffer_output.participant_id)

    def daemon_step(
            self,
            shared_memory_buffer_input: SharedMsgspecBufferWithEvent,
            shared_memory_buffer_output: SharedMsgspecBufferWithEvent,
            **kwargs  # pylint: disable=unused-argument
    ) -> None:
        """Computes the allowed tokens for the input token ids and schema."""
        shared_memory_buffer_input.wait_for_incoming_data()
        data: JSONLogitsProcessorInput = shared_memory_buffer_input.get_data()
        logger.info("Received data in JSONModeLogitsProcessor")
        shared_memory_buffer_input.clear()
        t = time.perf_counter()
        outputs = self.call(data.input_list)
        shared_memory_buffer_output.set_data(outputs)
        et = time.perf_counter() - t
        # THIS IS A TEMPORARY HACK TO AVOID MEMORY LEAKS
        # todo(yard1): Fix the lmformatenforcer itself to have
        # a bounded cache/come up with an alternate way of
        # handling this.
        memory_usage = self._process.memory_info().rss
        logger.info(
            "JSONModeLogitsProcessor call took %0.2fs, "
            "mem usage %d/%d", et, memory_usage, MAX_MEMORY_LIMIT)
        if self._process.memory_info().rss > MAX_MEMORY_LIMIT:
            logger.warning(
                "JSONModeLogitsProcessor memory usage %d "
                "exceeded limit %d, clearing cache", memory_usage,
                MAX_MEMORY_LIMIT)
            self.token_enforcer_cache.cache.clear()
            self.tensor_cache.cache.clear()
            gc.collect()

    def handle_step_exception(
            self,
            exception: Exception,  # pylint: disable=unused-argument
            shared_memory_buffer_input: SharedMsgspecBufferWithEvent,
            shared_memory_buffer_output: SharedMsgspecBufferWithEvent,
            **kwargs  # pylint: disable=unused-argument
    ) -> None:
        """Set the shared memory buffer to an error state."""
        shared_memory_buffer_input.set_error()
        shared_memory_buffer_output.set_error()


class JSONLogitsProcessorInputV2(msgspec.Struct, array_like=True):
    # token_ids, schema, request_id
    input_list: List[Tuple[List[int], str, str]]


SchemaCacheType = Tuple[GrammarEnforcer, Set[Tuple[int, ...]]]


class JSONModeLogitsProcessorV2(FaultAwareDaemon):

    def __init__(self,
                 rank: int,
                 tokenizer_name_or_path: str,
                 padded_vocab_size: int,
                 recreate_failed_actors: bool = False,
                 delay_between_actor_restarts_s: float = 0) -> None:
        super().__init__(recreate_failed_actors,
                         delay_between_actor_restarts_s)

        self.tokenizer = get_tokenizer(tokenizer_name_or_path)
        self.tokenizer_trie = TokenizerTrie(self.tokenizer,
                                            vocab_size=padded_vocab_size)
        self.json_mode_stack_prefix_cache = compute_json_mode_cache(
            self.tokenizer_trie)

        self.padded_vocab_size = padded_vocab_size
        self._rank = rank

        # Local cache from request_id to the token enforcer
        # And the set of list of tokens that have been accepted
        self.schema_grammar_cache: LRUCache[str, SchemaCacheType] = (
            LRUCache(SCHEMA_CACHE_SIZE))

    def daemon_setup(
            self,
            shared_memory_buffer_input: SharedMsgspecBufferWithEvent,
            shared_memory_buffer_output: SharedMsgspecBufferWithEvent,
            **kwargs  # pylint: disable=unused-argument
    ) -> None:
        """Setup the shared memory buffer for the JSON logits processor."""
        shared_memory_buffer_input.decoder = (
            msgspec.msgpack.Decoder(JSONLogitsProcessorInputV2))
        logger.info("JSON logits processor buffer id: "
                    "%d "
                    "%d", shared_memory_buffer_input.participant_id,
                    shared_memory_buffer_output.participant_id)

    def _create_enforcer_from_schema(self, schema: str) -> GrammarEnforcer:
        grammar = Grammar.from_json_schema(schema)
        logger.info("Grammar created successfully.")
        enforcer = GrammarEnforcer(
            grammar,
            self.tokenizer_trie,
            global_stack_prefix_cache=self.json_mode_stack_prefix_cache)
        enforcer.init()
        logger.info("Grammar enforcer initialized successfully.")
        return enforcer

    def call(self, data: JSONLogitsProcessorInputV2) -> np.ndarray:
        tokens_mask = np.zeros((len(data.input_list), self.padded_vocab_size),
                               dtype=np.bool_)
        for i, row in enumerate(data.input_list):
            token_ids, schema, request_id = row

            # Initialize accepted to True if there is no token_ids to be
            # accepted
            accepted = len(token_ids) == 0
            # Assumption is that if the enforcer is found locally,
            # it has been kept up-to-date.
            if request_id not in self.schema_grammar_cache:
                logger.info("request_id %s not found in the cache", request_id)
                enforcer = self._create_enforcer_from_schema(schema)
                logger.info("Accepting %d tokens ...", len(token_ids))
                prefix_states = set()
                for token_idx, token_id in enumerate(token_ids):
                    # Bring the state of the enforcer back to where
                    # it needs to be.
                    accepted = enforcer.accept_token(token_id)
                    prefix_states.add(tuple(token_ids[:token_idx + 1]))
                logger.info("Accepted tokens done.")
                self.schema_grammar_cache[request_id] = (enforcer,
                                                         prefix_states)
            else:
                logger.info("request_id %s found in the cache", request_id)
                enforcer, prefix_states = self.schema_grammar_cache[request_id]

                # Similar to v1, when there is a cache hit, we need to ensure
                # that all consecutive token_ids have been accepted in order.
                # So prefix_states is a local book-keeper that keeps track of
                # this.
                j = 0
                token_ids_tuple = tuple(token_ids)
                for j in range(len(token_ids), -1, -1):
                    if token_ids_tuple[:j] in prefix_states:
                        break
                for k in range(j, len(token_ids)):
                    accepted = enforcer.accept_token(token_ids[k])
                    prefix_states.add(token_ids_tuple[:k + 1])
                    if not accepted:
                        raise ValueError(
                            "Token `%d` not accepted by the "
                            "enforcer for some reason.", token_ids[-1])

            logger.info("Getting token mask ...")
            s = time.time()
            mask = enforcer.get_tokens_mask()
            time_get_tokens_mask_s = time.time() - s
            logger.info(
                "Got token mask in %0.2f s, "
                "mask.sum()=%s, "
                "eos_allowed=%s", time_get_tokens_mask_s, mask.sum(),
                mask[enforcer.tokenizer_trie.eos_token_id])
            tokens_mask[i, mask] = True
        return tokens_mask

    def daemon_step(
            self,
            shared_memory_buffer_input: SharedMsgspecBufferWithEvent,
            shared_memory_buffer_output: SharedMsgspecBufferWithEvent,
            **kwargs  # pylint: disable=unused-argument
    ) -> None:
        """Computes the allowed tokens for the input token ids and schema."""
        shared_memory_buffer_input.wait_for_incoming_data()
        data = shared_memory_buffer_input.get_data()
        data = cast(JSONLogitsProcessorInputV2, data)
        logger.info("Received data in JSONModeLogitsProcessor")
        shared_memory_buffer_input.clear()
        t = time.perf_counter()
        outputs = self.call(data)
        shared_memory_buffer_output.set_data(outputs)
        et = time.perf_counter() - t
        logger.info("JSONModeLogitsProcessor call took %0.2fs", et)

    def handle_step_exception(
            self,
            exception: Exception,  # pylint: disable=unused-argument
            shared_memory_buffer_input: SharedMsgspecBufferWithEvent,
            shared_memory_buffer_output: SharedMsgspecBufferWithEvent,
            **kwargs  # pylint: disable=unused-argument
    ) -> None:
        """Set the shared memory buffer to an error state."""
        shared_memory_buffer_input.set_error()
        shared_memory_buffer_output.set_error()
