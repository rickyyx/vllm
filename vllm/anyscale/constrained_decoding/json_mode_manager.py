import importlib
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import msgspec
import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from vllm.sequence import SequenceGroupMetadata

from vllm.anyscale.constrained_decoding.fault_tolerance import FaultAwareDaemon
from vllm.anyscale.constrained_decoding.logits_processor import (
    JSONLogitsProcessorInput, JSONLogitsProcessorInputV2,
    JSONModeLogitsProcessor, JSONModeLogitsProcessorV2)
from vllm.anyscale.shm.msgspec_shm import (RayEvent, SharedMemoryManager,
                                           SharedMemoryReadDataError,
                                           SharedMsgspecBufferWithEvent)
from vllm.anyscale.shm.numpy import numpy_encode_hook, numpy_ext_hook

MIN_ROWS_PER_JSON_LOGITS_PROCESSOR = int(
    os.getenv("ANYSCALE_VLLM_MIN_ROWS_PER_JSON_LOGITS_PROCESSOR", "1"))
JSON_LOGITS_PROCESSOR_TIMEOUT_S = int(
    os.getenv("ANYSCALE_VLLM_JSON_LOGITS_PROCESSOR_TIMEOUT_S", "20"))
LOGIT_PROCESSOR_ACTOR_RESTART_TIMEOUT_S = int(
    os.getenv("ANYSCALE_VLLM_LOGIT_PROCESSOR_ACTOR_RESTART_TIMEOUT_S", "20"))
LOGIT_PROCESSOR_PINGER_TIMEOUT_MS = int(
    os.getenv("ANYSCALE_VLLM_LOGIT_PROCESSOR_PINGER_TIMEOUT_MS", "10"))
SHARED_MEMORY_BUFFER_SIZE = int(
    os.getenv("ANYSCALE_VLLM_LOGIT_PROCESSOR_ACTOR_SHM_BUFFER_SIZE",
              "20_000_000"))  # 20 MB

logger = logging.getLogger(__name__)


@dataclass
class JSONLogitsProcessorPayload:

    # The index of the row matching the sequence
    logit_row_index: int
    # The output tokens for the sequence
    output_token_ids: List[int]
    # JSON schema for the sequence (shared across the group)
    # This can be any valid JSON schema
    schema: str

    # A unique identifier corresponding to this payload.
    # This is for example request_id from the server side.
    # This is a required field after json_mode v2.
    payload_id: Optional[str] = None

    @staticmethod
    def get_payload_id(seq_group_metadata: SequenceGroupMetadata,
                       seq_id: int) -> Optional[str]:
        """Returns a payload id given the input_metadata and seq_id."""
        req_id = seq_group_metadata.request_id
        if req_id:
            return f"{req_id}_{seq_id}"
        return None


class JSONModeManager:

    def __init__(self,
                 tokenizer_name_or_path: str,
                 vocab_size: int,
                 recreate_failed_actors: bool = True,
                 max_restarts: int = 5,
                 delay_between_actor_restarts_s: float = 0.0,
                 logit_processor_cls: Optional[Union[
                     str, Type[FaultAwareDaemon]]] = None,
                 use_v2: bool = False,
                 json_processor_num_workers: int = 1) -> None:
        """Initializes the json mode manager.

        Upon initialization, the manager will start the json logits
        processor workers loop and their shared memory buffers.

        Basically these workers are infinite loops that wait for a payload
        to be sent to them, process it and then send the result back to their
        shared memory buffer.

        Fault-tolerance behavior: If any of the logit processor actors dies due
        to any external reason (e.g. internal Exception, external oom killer,
        etc) instead of killing the engine which has down time in order of
        several minutes, we will restart the actor and continue monitoring it
        until it becomes healthy again. The batch of requests at which the
        logit processor was processing and it failed will be marked as failed
        and a custom error message will be send back to the user. In the
        meantime the logit_processing is offloaded to the rest of the healthy
        actors (assuming the failure is not universal among all of them). In
        case of a universal failure, we will wait for all actors to become
        healthy up to a timeout time and if nothing happens up to that timeout
        time, we will raise an error and kill the engine.

        NOTE: If any of the actors are unhealthy, because of the continuous
        monitoring, we will have a hit on decoding performance of the engine
        for requests with logit_processing. This can be solved if we notice
        this as a problem. Assuming that the frequency of the actor failures is
        low, addressing this is not a requirement because our baseline is a
        dead engine vs. lower performance replica for a few seconds until the
        actors become healthy again which is normally in the order of seconds.

        TODO (Kourosh): This behavior can be improved by improving the
        procedure for health check of actors to something that is done
        asynchronously and in parallel. Right now we control this by
        controlling the timeout time, if chosen too low, the ray actor will
        never get the time to communicate back that it is healthy, and the
        higher it is the higher the degradation of the perf (You are
        effectively waiting until this timeout time for the actor to respond
        to the ping-pong check).

        Args:
            tokenizer_name_or_path: The name or path of the tokenizer to use.
            vocab_size: The size of the vocabulary.
            recreate_failed_actors: If True, it restarts failed actor
                processes.
            max_restarts: Max number of actors to restart upon failures.
            delay_between_actor_restarts_s: Time it waits until it restarts
                new actors.
            logit_processor_cls: The logit processor class to use.
                JsonModeManager creates multiple replicas of a given logit
                processors. If it is passed via string, it imports the class
                from a given path.
            use_v2: Whether to use the v2 version of the json logits processor.
                This is the in-house version of the grammar enforcer. v1 is the
                version that is based on lm-format enforcer.
            json_processor_num_workers: The number of json processor workers to
                use. This is the number of actors that will be created to
                process the payloads.
        """

        self._use_v2 = use_v2
        logger.info("Use json mode v2: %s", self._use_v2)
        self._has_sent_payload = False

        default_logit_processor_cls = (JSONModeLogitsProcessorV2
                                       if use_v2 else JSONModeLogitsProcessor)

        # Import if logit_processor_cls is a string.
        if isinstance(logit_processor_cls, str):
            module_name, class_name = logit_processor_cls.rsplit(".", 1)
            module = importlib.import_module(module_name)
            logit_processor_cls = getattr(module, class_name)

        self.logit_processor_cls = (logit_processor_cls
                                    or default_logit_processor_cls)
        self._should_be_fault_tolerant = recreate_failed_actors

        # Logit Processor Actors and their input / output buffers
        self._json_mode_logits_processors = []
        self._logit_processor_daemon_input_buffers = []
        self._logit_processor_daemon_output_buffers = []

        # Logit processor actor ref -> Long running daemon task ref
        self._logit_processor_daemon_map = {}
        # Logit processor actor ref -> (input buffer, output buffer)
        self._actor_buffer_map: Dict[ray.ObjectRef,
                                     Tuple[SharedMsgspecBufferWithEvent,
                                           SharedMsgspecBufferWithEvent]] = {}

        # This is imported here to avoid circular imports.
        # deps: this -> model_executor -> models -> sampler -> this
        from vllm.distributed import (  # pylint: disable=import-outside-toplevel
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            model_parallel_is_initialized)

        use_tensor_parallelism = (model_parallel_is_initialized() and
                                  get_tensor_model_parallel_world_size() > 1)
        # Only keep actual processors on rank 0.
        if not use_tensor_parallelism or get_tensor_model_parallel_rank() == 0:
            current_node_scheduling_strategy = NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False)

            # Boilerplate for shared memory
            self._shared_mem_manager = SharedMemoryManager()
            self._shared_mem_manager.start()  # pylint: disable=consider-using-with

            self._shared_mem_event = RayEvent.options(
                num_cpus=0,
                scheduling_strategy=current_node_scheduling_strategy).remote()

            # Initialize shared memory buffers,
            # one input and one output for each processor worker
            self._logit_processor_daemon_input_buffers = [
                SharedMsgspecBufferWithEvent(
                    size=SHARED_MEMORY_BUFFER_SIZE,
                    manager=self._shared_mem_manager,
                    encoder_init_fn=lambda: msgspec.msgpack.Encoder(
                        enc_hook=numpy_encode_hook),
                    decoder_init_fn=lambda: msgspec.msgpack.Decoder(
                        ext_hook=numpy_ext_hook),
                    ray_event=self._shared_mem_event,
                ) for _ in range(json_processor_num_workers)
            ]
            self._logit_processor_daemon_output_buffers = [
                SharedMsgspecBufferWithEvent(
                    size=SHARED_MEMORY_BUFFER_SIZE,
                    manager=self._shared_mem_manager,
                    encoder_init_fn=lambda: msgspec.msgpack.Encoder(
                        enc_hook=numpy_encode_hook),
                    decoder_init_fn=lambda: msgspec.msgpack.Decoder(
                        ext_hook=numpy_ext_hook),
                    ray_event=self._shared_mem_event,
                ) for _ in range(json_processor_num_workers)
            ]
            input_buffer_ids = [
                buffer.participant_id
                for buffer in self._logit_processor_daemon_input_buffers
            ]
            output_buffer_ids = [
                buffer.participant_id
                for buffer in self._logit_processor_daemon_output_buffers
            ]
            logger.info(
                "Sampler shared memory buffer ids. "
                "input: %s output: %s", input_buffer_ids, output_buffer_ids)

            extra_ray_kwargs = {}
            if recreate_failed_actors:
                extra_ray_kwargs = {
                    "max_restarts": max_restarts,
                }

            remote_cls = ray.remote(self.logit_processor_cls).options(
                num_cpus=0,
                scheduling_strategy=current_node_scheduling_strategy,
                **extra_ray_kwargs,
            )
            self._json_mode_logits_processors = [
                remote_cls.remote(
                    i,
                    tokenizer_name_or_path,
                    vocab_size,
                    recreate_failed_actors=recreate_failed_actors,
                    delay_between_actor_restarts_s=(
                        delay_between_actor_restarts_s))
                for i in range(json_processor_num_workers)
            ]

            # Wait until the actors are ready
            creation_s = time.time()
            ray.get([
                actor.__ray_ready__.remote()
                for actor in self._json_mode_logits_processors
            ])
            logger.info("Logits processors got created "
                        "in %.2fs",
                        time.time() - creation_s)

            # zip(x, y) would silently not work as expected if x and y
            # don't have the same length.
            assert len(self._json_mode_logits_processors) == len(
                self._logit_processor_daemon_input_buffers)
            assert len(self._json_mode_logits_processors) == len(
                self._logit_processor_daemon_output_buffers)
            # Start infinite loops inside the processor workers
            # With fault-aware apply when the actor dies the error will be
            # caught and it will get restarted.
            for processor, input_buffer, output_buffer in zip(
                    self._json_mode_logits_processors,
                    self._logit_processor_daemon_input_buffers,
                    self._logit_processor_daemon_output_buffers):
                daemon_ref = processor.run.remote(input_buffer, output_buffer)
                self._logit_processor_daemon_map[processor] = daemon_ref
                self._actor_buffer_map[processor] = (input_buffer,
                                                     output_buffer)

    def _restart_unhealthy_actors(self,
                                  actor_indices: Set[int],
                                  blocking: bool = False) -> Set[int]:
        """Restarts the unhealthy actors.

        This method is called when an actor has been restarted but has not come
        back yet. This will wait until all actors are healthy again. During the
        process it will also clear the state of the shared memory buffers and
        restarts the daemon processes.

        Args:
            actor_indices: The indices of the actors to restart.
            blocking: Whether to block until the actors are healthy.
        Returns:
            The indices of the actors that have been restarted.
        """
        logger.info("Restarting unhealthy actors...")

        restarted_buffer_inds = set()
        for actor_index in actor_indices:
            actor = self._json_mode_logits_processors[actor_index]

            # Actor is restarted or left in a bad state. wait until timeout (if
            # blocking) to recover
            logger.info("Waiting on actor %s to become healthy ...", actor)
            s_time = time.time()
            resp = ""
            while True:
                try:
                    resp = ray.get(actor.ping.remote(),
                                   timeout=LOGIT_PROCESSOR_PINGER_TIMEOUT_MS /
                                   1000.0)
                    break
                except (ray.exceptions.RayActorError,
                        ray.exceptions.GetTimeoutError) as e:
                    if blocking:
                        if ((time.time() - s_time) >
                                LOGIT_PROCESSOR_ACTOR_RESTART_TIMEOUT_S):
                            raise TimeoutError(
                                f"Actor {actor} did not become healthy in time."
                            ) from e
                    else:
                        break

            logger.info("The restart attempt for actor %s took"
                        "%d seconds.", actor,
                        time.time() - s_time)

            # Move on to the next unhealthy actor if the actor is not ready
            if resp != "pong":
                continue

            logger.info("Actor %s is healthy, clearing the buffer ...", actor)

            input_buffer, output_buffer = self._actor_buffer_map[actor]
            input_buffer.clear_error()
            input_buffer.clear()
            output_buffer.clear_error()
            output_buffer.clear()

            logger.info(
                "Actor %s is healthy and buffers are now clean,"
                "restarting the daemon ...", actor)
            self._logit_processor_daemon_map[actor] = actor.run.remote(
                input_buffer, output_buffer)

            restarted_buffer_inds.add(actor_index)

            logger.info("Actor %s restarted successfully.", actor)

        return restarted_buffer_inds

    def _restart_unhealthy_actors_and_get_healthy_buffers(self) -> List[int]:
        """Returns a list of indices of the healthy actors and buffers.

        This function also restarts the unhealthy actors and clears the shared
        memory buffer associated with them.
        """

        # If any of the actors is unhealthy we continue on the remaining
        # healthy actors and adjust the workload assigned to each one
        # Note the error will be set on both input and output buffers.
        all_buffer_inds = set(
            range(len(self._logit_processor_daemon_input_buffers)))
        unhealthy_buffer_inds = set()
        for buffer_index, buffer in enumerate(
                self._logit_processor_daemon_input_buffers):
            if buffer.has_errored():
                unhealthy_buffer_inds.add(buffer_index)
        healthy_buffer_inds = all_buffer_inds - unhealthy_buffer_inds

        if unhealthy_buffer_inds:
            # We only block if there is no healthy daemon process
            # We are assuming that the underlying actors fail rarely. If that
            # is a wrong assumption we should find a better way to handle this
            # and not block the engine for a long time.

            # Log healthy actors when unhealthy actors are detected so that we
            # can find them during an unhealthy state.
            healthy_actors = {
                self._json_mode_logits_processors[idx]
                for idx in healthy_buffer_inds
            }
            logger.info("The current healthy actors are %s", healthy_actors)

            blocking = len(healthy_buffer_inds) == 0
            restarted_buffer_inds = self._restart_unhealthy_actors(
                actor_indices=unhealthy_buffer_inds, blocking=blocking)
            if restarted_buffer_inds:
                healthy_buffer_inds = healthy_buffer_inds.union(
                    restarted_buffer_inds)

        # Sort the healthy buffers to maximize the chance of colocation of the
        # similar schemas (payloads are sorted by schema)
        ordered_healthy_buffer_inds = sorted(list(healthy_buffer_inds))

        return ordered_healthy_buffer_inds

    def start_json_logits_processors(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> Dict[int, List[int]]:
        """Start JSON processors by sending them new data. Non-blocking.

        See `Sampler.start_json_logits_processors` for more information.

        Fault-tolerance behavior: If a dead actor is detected it will pause
        until that actor is healthy again (up to some timeout time).
        """
        # buffer_inds_to_batch_inds -> allows us to later match
        # returned data back to the logits tensor
        buffer_inds_to_batch_inds = {}
        payloads = self._get_processor_payloads(seq_group_metadata_list)

        healthy_buffer_inds = (
            self._restart_unhealthy_actors_and_get_healthy_buffers())

        # Get how many processors we will use
        n_processors = min(
            math.ceil(len(payloads) / MIN_ROWS_PER_JSON_LOGITS_PROCESSOR),
            len(self._json_mode_logits_processors),
            len(healthy_buffer_inds),
        )

        # If we have any payload, at least one request in the batch is in
        # json-mode sampling
        if payloads and n_processors > 0:

            set_event = True
            indices = range(len(payloads))

            # Chunk the data equally between the processors
            for healthy_processor_idx, processor_payload_indices in enumerate(
                    np.array_split(indices, n_processors)):
                # What indices of the data list the processor will have
                processor_payload_indices = processor_payload_indices.tolist()
                # Get the chunk according to the indices
                s_idx = processor_payload_indices[0]
                e_idx = processor_payload_indices[-1] + 1
                payload_chunk = payloads[s_idx:e_idx]

                buf_idx = healthy_buffer_inds[healthy_processor_idx]
                buffer_inds_to_batch_inds[buf_idx] = ([
                    p.logit_row_index for p in payload_chunk
                ])

                processor_input = self._convert_payloads_to_processor_input(
                    payload_chunk)

                # Send the data to the processors by writing to the
                # shared memory
                if self._json_mode_logits_processors:
                    shm = self._logit_processor_daemon_input_buffers[buf_idx]
                    shm.wake_up(set_event=set_event)
                    shm.set_data(processor_input, wake_up=False)
                    set_event = False
            self._has_sent_payload = True
        return buffer_inds_to_batch_inds

    def get_json_logits_bias(
        self,
        logits: torch.Tensor,
        buffer_inds_to_batch_inds: Dict[int, List[int]],
    ) -> Tuple[torch.Tensor, List[bool]]:
        """Get the logit bias tensors from the json logit processors.

        This tensor will be used to mask the logits tensor. It will be 0 for
        those token_ids that are allowed and -inf for those that are not.

        Args:
            logits: The raw logits tensor from the model.
            buffer_inds_to_batch_inds: The mapping from buffer indices to batch
                indices.

        Returns:
            A tuple where the first element is the mask tensor. 
            The second element is a list of bools which indicates which
            rows are valid masks. If no guided requests are in the batch 
            the mask will be all ones.
        """
        # If we have any sequences with json enabled...
        if buffer_inds_to_batch_inds:
            # Only get here if we have pending logit processor actors and
            # non-empty buffer_inds_to_batch_inds (rank 0 only)
            logits_bias = torch.full_like(logits, float("-inf"))
            all_allowed_tokens_mask, mask_success = (
                self._get_mask_from_processor_output(
                    logits, buffer_inds_to_batch_inds))
            # Apply the generated mask
            all_allowed_tokens_mask = all_allowed_tokens_mask.to(
                device=logits.device, non_blocking=True)
            logits_bias.masked_fill_(all_allowed_tokens_mask, 0)
        else:
            # On non rank-0 and non-empty payload or if guided payload is empty,
            # we just create no-op tensors, which we'll also broadcast into
            # from rank 0.
            logits_bias = torch.zeros_like(logits)
            mask_success = [True] * len(logits)

        return logits_bias, mask_success

    def _convert_payloads_to_processor_input(
        self, payloads: List[JSONLogitsProcessorPayload]
    ) -> Union[JSONLogitsProcessorInput, JSONLogitsProcessorInputV2]:
        """Converts the payloads to the input for the json logits processors."""

        if self._use_v2:
            input_list = []
            for payload in payloads:
                input_list.append((payload.output_token_ids, payload.schema,
                                   payload.payload_id))

            return JSONLogitsProcessorInputV2(input_list=input_list)

        return JSONLogitsProcessorInput(input_list=[(
            p.output_token_ids,
            p.schema,
        ) for p in payloads])

    def _get_processor_payloads(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> List[JSONLogitsProcessorPayload]:
        """Get the payload to send to the json logits processors.

        Convert the input into per-sequence payload to send to the
        processor workers. Payload is a unit of data that is sent to the json
        processor workers to compute the mask for the logits. We will then
        distribute these payloads to the workers in equal chunks via the shared
        memory buffers.

        If we sort the payloads by schema, then we will increase the cache hit
        rate for the workers, as they will be able to reuse the schema data
        from the previous call.

        Args:
            seq_group_metadata_list: A list of sequence group to generate
                per-sequence payload to send to the json processor worker.

        Returns:
            A list of payloads to send to the json processor workers.
        """
        logits_row_idx = 0
        payloads: List[JSONLogitsProcessorPayload] = []
        # this will be rewritten to not use ray.
        # for each sequence, if there is a corresponding
        # json_mode_logits_processor, call it with the sequence's
        # output_token_ids. The call is parallelized remotely via ray since
        # they are long (> 1 ms).

        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            # TODO(sang): Prompt logprob is not supported.
            assert sampling_params.prompt_logprobs is None, (
                "Prompt logprob is not supported with json mode.")
            seq_ids = list(seq_group_metadata.seq_data.keys())
            # It is already validated int SamplingParam's postint that
            # response_format is a dict with a schema key
            if sampling_params.response_format:
                json_schema = sampling_params.response_format["schema"]

                # We need to prepare our data.
                # Each sequence group can contain multiple sequences.
                # We are operating on a per-sequence basis.
                for i, seq_id in enumerate(seq_ids):
                    seq_data = seq_group_metadata.seq_data[seq_id]
                    output_tokens = seq_data.get_output_token_ids()

                    payload_id = JSONLogitsProcessorPayload.get_payload_id(
                        seq_group_metadata, seq_id=seq_id)

                    payloads.append(
                        JSONLogitsProcessorPayload(
                            logit_row_index=logits_row_idx + i,
                            output_token_ids=output_tokens,
                            schema=json_schema,
                            payload_id=payload_id))

            # Advance the row index by the length of sequence
            # TODO(sang): This is not working with chunked prefill yet.
            logits_row_idx += len(seq_ids)

        # Sort by schema to colocate as much as possible
        payloads.sort(key=lambda x: x.schema)
        return payloads

    def _get_mask_from_processor_output(
        self,
        logits: torch.Tensor,
        buffer_inds_to_batch_inds: Dict[int, List[int]],
    ) -> Tuple[torch.Tensor, List[bool]]:
        """Constructs a boolean mask from the output of processors.

        It reads the shared memory buffers and re-indexes the output to match
        the structure of the logits. Then it sets the mask to 1 for the allowed
        tokens and 0 for the rest.

        This function also clears the shared memory buffers and sets the
        `_has_sent_payload` flag to False.

        Fault-tolerance behavior: If there is an error (dead or hanging actor),
        we will mark the allocated batch to be invalid and continue with no
        masking for those batch items. The sampler will then will mark them as
        invalid as well.

        Args:
            logits: The raw logits tensor from the model.
            buffer_inds_to_batch_inds: The mapping from buffer indices to batch
                indices.

        Returns:
            A tuple where the first element is the mask tensor. 
            The second element is a list of bools which indicates which
            rows are valid masks.
        """
        # Should not be called on rank != 0
        assert self._json_mode_logits_processors
        # We needed to have sent a payload before
        assert self._has_sent_payload

        all_allowed_tokens_mask = torch.ones(logits.shape,
                                             dtype=torch.bool,
                                             pin_memory=True)

        valid_mask = [True] * len(logits)

        for buf_idx, batch_inds in buffer_inds_to_batch_inds.items():
            # Get returned data from a single processor
            o_buffer = self._logit_processor_daemon_output_buffers[buf_idx]
            i_buffer = self._logit_processor_daemon_input_buffers[buf_idx]

            try:
                o_buffer.wait_for_incoming_data(
                    timeout_s=JSON_LOGITS_PROCESSOR_TIMEOUT_S)
            except TimeoutError as e:
                # If buffer has error or data we won't hit the timeout, if we
                # hit timeout it means that the actor is stuck or dead due to
                # some other reason that doesn't set the error flag on the
                # buffer. We will kill the actor in case it is stuck in the
                # daemon process with restart.
                if not self._should_be_fault_tolerant:
                    raise e
                logger.info(
                    "Hit the timeout error in mask_from_processor_output,"
                    "Killing the actor to ensure a fresh start ...")
                ray.kill(self._json_mode_logits_processors[buf_idx],
                         no_restart=False)
                logger.info("Setting the buffer state to error ...")
                i_buffer.set_error()
                o_buffer.set_error()
                logger.info("Continuing to the next buffer ...")

                for batch_idx in batch_inds:
                    valid_mask[batch_idx] = False
                continue

            try:
                allowed_token_mask = o_buffer.get_data()
            except SharedMemoryReadDataError as e:
                if not self._should_be_fault_tolerant:
                    raise e
                # In this case we will just continue and the
                # start_json_logits_processors will take care of restarting and
                # clearing the buffer
                logger.info(
                    "Hit the shm read error in mask_from_processor_output ...")
                for batch_idx in batch_inds:
                    valid_mask[batch_idx] = False
                continue

            allowed_token_mask_tensor = torch.from_numpy(allowed_token_mask)

            # Reindex
            all_allowed_tokens_mask[batch_inds] = allowed_token_mask_tensor

            o_buffer.clear()

        self._has_sent_payload = False

        return all_allowed_tokens_mask, valid_mask
