"""A layer that compute logits from hidden_stats."""
import inspect
from typing import List, Optional

import torch
import torch.nn as nn

from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SequenceGroupMetadata

from vllm.anyscale import anyscale_envs
from vllm.anyscale.constrained_decoding.json_mode_manager import (
    JSONModeManager)
from vllm.platforms import current_platform


class LogitsProcessor(nn.Module):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_gather = not current_platform.is_tpu()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.logits_as_input:
            logits = hidden_states
        else:
            hidden_states = _prune_hidden_states(hidden_states,
                                                 sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
            logits = _apply_logits_processors(logits, sampling_metadata)

        return logits

    def _get_logits(self, hidden_states: torch.Tensor,
                    lm_head: VocabParallelEmbedding,
                    embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = lm_head.linear_method.apply(lm_head,
                                             hidden_states,
                                             bias=embedding_bias)
        if self.use_gather:
            logits = tensor_model_parallel_gather(logits)
        else:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", forg_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    return hidden_states.index_select(0,
                                      sampling_metadata.selected_token_indices)


def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    found_logits_processors = False
    logits_processed = 0
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True

            for seq_id, logits_row_idx in zip(seq_ids,
                                              seq_group.sample_indices):
                logits_row = logits[logits_row_idx]
                past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids

                for logits_processor in logits_processors:
                    parameters = inspect.signature(logits_processor).parameters
                    if len(parameters) == 3:
                        logits_row = logits_processor(prompt_tokens_ids,
                                                      past_tokens_ids,
                                                      logits_row)
                    else:
                        logits_row = logits_processor(past_tokens_ids,
                                                      logits_row)

                logits[logits_row_idx] = logits_row

        logits_processed += len(seq_group.sample_indices) + len(
            seq_group.prompt_logprob_indices)

    if found_logits_processors:
        # verifies that no rows in logits were missed unexpectedly
        assert logits_processed == logits.shape[0]

    return logits


# Anyscale start
class JsonModeAsyncBatchLogitProcessor(LogitsProcessor):
    """Logit processor that is used for json mode.

    Compared to a normal guided decoding, json logit masks are
    prepared "asynchronously" in a batch. Note that a normal logit processor
    prepares mask "synchronously" row by row blocking.

    TODO(sang): This class should be removed once we support batch logit
    processor in OSS vLLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mapping from buffer indicies to the batch indices that the
        # corresponding buffer is responsible for. Since vLLM only allows 1
        # batch at a time to be processed, it is set to None when logit masks
        # are not prepared by `prepare`.
        self._buffer_inds_to_batch_inds = None
        # Lazily initialized.
        self._json_mode_manager: Optional[JSONModeManager] = None

        self._copy_stream: Optional[torch.cuda.Stream] = None

    def initialize(self, tokenizer: str) -> None:
        """Initialize the stateful logit processors.

        Args:
            tokenizer_name_or_path: The name or path of the tokenizer to use.
        """
        if self._json_mode_manager is not None:
            return

        self._json_mode_manager = JSONModeManager(
            tokenizer,
            self.vocab_size,
            recreate_failed_actors=anyscale_envs.RECREATE_FAILED_ACTORS,
            max_restarts=anyscale_envs.MAX_RESTARTS,
            delay_between_actor_restarts_s=anyscale_envs.
            DELAY_BETWEEN_ACTOR_RESTARTS_S,
            logit_processor_cls=anyscale_envs.LOGIT_PROCESSOR_CLS,
            use_v2=anyscale_envs.USE_V2,
            json_processor_num_workers=anyscale_envs.NUM_PROCESSOR_WORKERS,
        )

        self._copy_stream = torch.cuda.Stream()

    def prepare(self,
                seq_gruop_metadata_list: List[SequenceGroupMetadata]) -> None:
        """Prepare logit masks by start JSON processors by sending them
            new data. Non-blocking.

        This method submits the batch items that need to be logit processed in
        a non-blocking fashion. It also sets the batch indices mapping so
        that we can recover the processed logits and assign them back to the
        item indices.

        Args:
            seq_gruop_metadata_list: A list of sequence group metadata to
                prepare logit masks for json mode. 
        """
        assert self._json_mode_manager is not None
        self._buffer_inds_to_batch_inds = (
            self._json_mode_manager.start_json_logits_processors(
                seq_gruop_metadata_list))

    def _apply_json_logits_processor(
        self,
        logits: torch.Tensor,
    ) -> Optional[List[bool]]:
        """Apply the JSON processor mask to logits in-place.

        `prepare` has to be called before this API used. After this API
        is called the state prepared from `prepared` it reset.

        Args:
            logits: The logits tensor to apply the mask to.
                This tensor will be modified in-place.

        Returns:
            A list of bools indicating which sequences finished successfully.
        """
        assert (self._json_mode_manager is not None or self._copy_stream
                is not None), (".initialize should have been called already, "
                               "Did you forget it?")
        # `prepare` has to be called before forward is called.
        assert self._buffer_inds_to_batch_inds is not None, (
            ".prepare(..) should have been called, Did you forget it?")

        with torch.cuda.stream(self._copy_stream):
            json_bias, json_mask_success = (
                self._json_mode_manager.get_json_logits_bias(  # type: ignore
                    logits, self._buffer_inds_to_batch_inds))
        torch.cuda.current_stream().wait_stream(self._copy_stream)

        logits.add_(json_bias)
        self._buffer_inds_to_batch_inds = None

        return json_mask_success

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits and apply the json logit processor.

        `prepare` has to be called before `forward` is called. And `prepare`
        and `forward` should be called 1:1. Otherwise, it will have assertion
        failure.
        """
        logits = super().forward(
            embedding,
            hidden_states,
            sampling_metadata,
            embedding_bias,
        )

        # Non-driver worker can return None for logits.
        if logits is None:
            return logits, None

        json_mask_success = self._apply_json_logits_processor(logits)

        return logits, json_mask_success


if anyscale_envs.ENABLE_JSON_MODE:
    LogitsProcessor = JsonModeAsyncBatchLogitProcessor  # type: ignore
# Anyscale end
