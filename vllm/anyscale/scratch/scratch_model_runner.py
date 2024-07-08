import dataclasses
import importlib.util
import sys
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Set,
                    Type, TypeVar, Union)

import boto3
import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SamplerOutput, SequenceGroupMetadata,
                           SequenceOutput)
from vllm.utils import CudaMemoryProfiler, LRUCache, is_pin_memory_available
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

from vllm.anyscale.anyscale_envs import USE_SCRATCH, USE_SCRATCH_SAMPLE
from vllm.anyscale.scratch.constants import (SCRATCH_EXECUTABLE_PATH,
                                             SCRATCH_TMP_DIR,
                                             SCRATCH_WEIGHTS_BUCKET_NAME,
                                             SCRATCH_WEIGHTS_PREFIX)

logger = init_logger(__name__)


def import_scratch(path: Path):
    SCRATCH_MODULE_NAME = "scratch"
    logger.info("Importing scratch module from %s", path)
    spec = importlib.util.spec_from_file_location(SCRATCH_MODULE_NAME,
                                                  path.resolve())
    scratch = importlib.util.module_from_spec(spec)
    sys.modules[SCRATCH_MODULE_NAME] = scratch
    spec.loader.exec_module(scratch)
    return scratch


scratch_mod = import_scratch(Path(SCRATCH_EXECUTABLE_PATH))


class ScratchSession:
    """An abstraction to store Scratch session."""

    def __init__(self, scratch_session_id: int):
        self.scratch_session_id = scratch_session_id


class ScratchLRUCache(LRUCache[ScratchSession]):
    """LRU cache to store scratch sessions.

    It is a temporary hack to figure out the finished sessions.
    It relies on vllm guarantees to complete running seq groups.
    """

    def __init__(self, capacity: int, scratch_api):
        self._scratch_api = scratch_api
        super().__init__(capacity)

    def _on_remove(self, key: Hashable, value: ScratchSession):
        # Currently, key and values are both int session id.
        self._scratch_api.delete_session(value.scratch_session_id)


class ScratchSessionManager:
    """A class that manages multiple scratch sessions.

    Stale sessions are currently automatically deleted. "IT IS A HACK".
    vLLM model runner cannot know which sequence groups are finished/preempted.
    We use LRUCache to track the max_num_seqs * 2 session and clean up session
    that are least recently used. (Note this also means we can use up to 2X
    GPU memory for kv cache than needed). It is working because vLLM scheduler
    prioritizes running decode all the time. It may break when preemption of
    swapping happens. Currently, we are disabling these features when
    ScratchLLM is used. This can be solved once we pipeline the
    finished/preempted sequence group information to model runner in a few
    weeks.
    """

    def __init__(self, scratch_api, max_num_seqs: int):
        # ScratchAPI used to create/delete sessions.
        self._scratch_api = scratch_api
        # Set capacity to max_num_seqs * 2 so that old sequences are
        # deleted. Note that it is a hack, and it will be fixed once
        # vLLM plumbs finished session info to the model runner.
        # vLLM request_id -> scratch session.
        self._session_cache = ScratchLRUCache(max_num_seqs * 2, scratch_api)

    def get_or_create_session(self, vllm_request_id: int) -> ScratchSession:
        if vllm_request_id in self._session_cache:
            return self._session_cache[vllm_request_id]

        # Session id is guaranteed to be unique from scratch.
        scratch_session_id: int = self._scratch_api.new_session()
        session = ScratchSession(scratch_session_id)
        self._session_cache[vllm_request_id] = session
        return session


TModelInputForScratch = TypeVar('TModelInputForScratch',
                                bound="ModelInputForScratch")


@dataclasses.dataclass(frozen=True)
class ModelInputForScratch(ModelRunnerInputBase):
    input_tokens: List[int]
    parent_ids: List[int]
    session_ids: List[int]
    prefill_groups: List[SequenceGroupMetadata]
    decode_groups: List[SequenceGroupMetadata]
    query_lens: List[int]
    seq_lens: List[int]
    sampling_metadata: SamplingMetadata

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("TP>1 is not supported")

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForScratch],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForScratch:
        raise NotImplementedError("TP>1 is not supported")


class ScratchModelRunner(ModelRunnerBase[ModelInputForScratch]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        multimodal_config: Optional[MultiModalConfig] = None,
        return_hidden_states: bool = False,
    ):
        assert USE_SCRATCH, (
            "Use ANYSCALE_VLLM_USE_SCRATCH_LLM=1 to use ScratchLLM")

        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.cache_config = cache_config
        self.is_driver_worker = is_driver_worker
        self.multimodal_config = multimodal_config

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.lora_manager = None
        self.model_config.enforce_eager = True
        self.kv_cache_dtype = kv_cache_dtype
        self.pin_memory = is_pin_memory_available()

        # Lazily initialized.
        self.scratch: scratch_mod.ScratchAPI  # type: ignore
        # We create a model that has a subset of functionalities
        # that scratch doesn't support. I.e., norm, logit compute,
        # and sampling. NOTE: We should make sure the config is in sync.
        self.model: nn.Module
        self._scratch_session_manager: ScratchSessionManager
        self._verify_scratch_config()

    def _verify_scratch_config(self):
        # TODO(sang): We will need better config validation than this.
        # Ideally, we should choose a subset of config supported by
        # scratch from config.py or arg_utils.py.
        assert self.is_driver_worker, ("TP > 1 not supported.")
        assert self.model_config.dtype == torch.half, (
            "Only half type is allowed.")
        assert self.scheduler_config.chunked_prefill_enabled is False, (
            "Chunked prefill not supported")
        assert self.sliding_window is None, ("Sliding window not supported")
        assert self.kv_cache_dtype == "auto", (
            "Currently, Scratch doesn't use kv cache.")
        assert ("llama-2" in self.model_config.model.lower()
                or "llama-3" in self.model_config.model.lower()), (
                    "Only Llama 2 7B or llama 3 8B is supported.")
        assert self.lora_manager is None, ("lora is not supported.")
        assert self.model_config.enforce_eager is True, (
            "cuda graph is not needed for Scratch.")
        assert self.cache_config.block_size == 32, (
            "Scratch only supports page size of 32.")
        assert self.cache_config.enable_prefix_caching is False, (
            "enable_prefix_caching is not supported for Scratch")

    def load_model(self) -> None:
        assert self.load_config.download_dir is None
        tmp_dir = Path(SCRATCH_TMP_DIR)
        tmp_dir.mkdir(exist_ok=True)
        weights_dir = tmp_dir / "parameters"
        weights_dir.mkdir(exist_ok=True)
        # TODO(sang): Need to obtain this programmatically.
        # download_dir = weights_dir / "ll27b-s1-cuda-f16-fullopt"
        base_dir = str(weights_dir.resolve())
        self.scratch = scratch_mod.ScratchAPI(base_dir)
        scratch_subdir = self.scratch.get_param_subdir()
        download_dir = weights_dir / scratch_subdir
        download_dir.mkdir(exist_ok=True)
        download_dir_path = str(download_dir.absolute())
        self.load_config.download_dir = str(weights_dir.absolute())
        self._download_scratch_weights(SCRATCH_WEIGHTS_PREFIX,
                                       download_dir_path,
                                       SCRATCH_WEIGHTS_BUCKET_NAME)

        with CudaMemoryProfiler() as m:
            self.model = get_model(
                model_config=self.model_config,
                device_config=self.device_config,
                load_config=self.load_config,
                lora_config=self.lora_config,
                multimodal_config=self.multimodal_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                cache_config=self.cache_config,
            )
            self.scratch.start()
            self._scratch_session_manager = ScratchSessionManager(
                self.scratch, self.scheduler_config.max_num_seqs)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        # KV cache dtype/quantization is not supported.

    def _download_scratch_weights(self, prefix: str, target_dir: str,
                                  bucket: str):
        # TODO(sang): Use fast loading.
        s3_client = boto3.client('s3')
        files: List[str] = []
        dirs: List[str] = []
        next_token = ""
        base_kwargs = {"Bucket": bucket, "Prefix": prefix}
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != "":
                kwargs.update({"ContinuationToken": next_token})
            results = s3_client.list_objects_v2(**kwargs)
            contents = results.get("Contents")
            for content in contents:
                k = content.get("Key")
                if k[-1] != "/":
                    files.append(k)
                else:
                    dirs.append(k)
            next_token = results.get('NextContinuationToken')
        # Assume there's no subdirectories.
        dirs = {p.rsplit("/", 1)[0] for p in files}
        assert len(dirs) == 1, dirs

        # NOTE(sang): Versioning is not supported now. We assume the
        # weights are always the same.
        # NOTE: Threadpool doesn't really improve performance.
        # Maybe it is rate limited.
        for file in tqdm(files,
                         desc=f"Downloading scratch weights to {target_dir}"):
            dest = Path(target_dir) / Path(file).name
            if not dest.exists():
                s3_client.download_file(bucket, file, str(dest.absolute()))

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForScratch:
        input_tokens: List[int] = []
        parent_ids: List[int] = []
        session_ids: List[int] = []
        prefill_groups: List[SequenceGroupMetadata] = []
        decode_groups: List[SequenceGroupMetadata] = []
        query_lens: List[int] = []
        seq_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            session_ids.append(
                self._scratch_session_manager.get_or_create_session(
                    request_id).scratch_session_id)

            is_prefill = seq_group_metadata.is_prompt
            if is_prefill:
                prefill_groups.append(seq_group_metadata)
            else:
                decode_groups.append(seq_group_metadata)

            seq_data = seq_group_metadata.seq_data
            for seq_id, data in seq_data.items():
                # It is the case only when num_seq == 1, i.e., beam search
                # it not used.
                parent_id = seq_id
                parent_ids.append(parent_id)
                if is_prefill:
                    input_tokens.extend(data.prompt_token_ids)
                    prompt_len = seq_data[parent_id].get_prompt_len()
                    query_lens.append(prompt_len)
                    seq_lens.append(prompt_len)
                else:
                    input_tokens.append(data.get_last_token_id())
                    query_lens.append(1)
                    seq_lens.append(seq_data[parent_id].get_len())

        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     seq_lens, query_lens,
                                                     self.device,
                                                     self.pin_memory)
        return ModelInputForScratch(
            input_tokens=input_tokens,
            parent_ids=parent_ids,
            session_ids=session_ids,
            prefill_groups=prefill_groups,
            decode_groups=decode_groups,
            query_lens=query_lens,
            seq_lens=seq_lens,
            sampling_metadata=sampling_metadata,
        )

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForScratch:
        raise NotImplementedError("TP > 1 is not supported.")

    def set_block_size(self, block_size: int) -> None:
        # It will be relevant later.
        self.block_size = block_size

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForScratch,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        # KV cache is unused.
        assert num_steps == 1

        if USE_SCRATCH_SAMPLE:
            return [
                self._execute_and_scratch_sample(model_input.prefill_groups,
                                                 model_input.decode_groups,
                                                 model_input.input_tokens,
                                                 model_input.session_ids,
                                                 model_input.parent_ids,
                                                 model_input.query_lens)
            ]
        else:
            return [
                self._execute_and_vllm_sample(model_input.prefill_groups,
                                              model_input.decode_groups,
                                              model_input.input_tokens,
                                              model_input.session_ids,
                                              model_input.query_lens,
                                              model_input.sampling_metadata)
            ]

    def _execute_and_vllm_sample(
        self,
        prefill_groups: List[SequenceGroupMetadata],
        decode_groups: List[SequenceGroupMetadata],
        input_tokens: List[int],
        session_ids: List[int],
        query_lens: List[int],
        sampling_metadata: SamplingMetadata,
    ):
        """Run scratchLLM kernels with vLLM logit processor + sampler.

        Args:
            prefill_groups: A list of sequence group metadata for prefill
            decode_groups: A list of sequence group metadata for decode
            input_tokens: (num_batched_tokens) input tokens.
            session_ids: A list of session ids.
            query_lens: (num_seqs). A length of input tokens per sequence.
                Used to find the length of input queries from 1D
                `input_tokens`.
            sampling_metadata: SamplingMetadata used for sampling.

        Returns:
            SamplerOutput for a given batch of requests.
        """
        if len(prefill_groups) > 0:
            assert len(decode_groups) == 0
        if len(decode_groups) > 0:
            assert len(prefill_groups) == 0

        batch_size = len(session_ids)
        hidden_size = self.model_config.get_hidden_size()
        hidden_states = torch.empty(len(input_tokens) * hidden_size,
                                    device=self.device,
                                    dtype=torch.half)
        input_tokens_tensor = torch.tensor(input_tokens,
                                           device=self.device,
                                           dtype=torch.int)

        # Run prefills. Scratch currently doesn't support batch prefills,
        # so we should iterate one by one.
        cum_query_length = 0
        for i, session_id in enumerate(session_ids):
            # Find relevant tensor from 1D input token tensors.
            query_len = query_lens[i]
            prefill_req_tensor = input_tokens_tensor[
                cum_query_length:cum_query_length + query_len]
            # Hidden state should multiply hidden_size because it is flattened.
            hidden_state = hidden_states[cum_query_length *
                                         hidden_size:(cum_query_length +
                                                      query_len) * hidden_size]

            self.scratch.prefill(
                session_id,
                prefill_req_tensor.data_ptr(),
                prefill_req_tensor.shape[0],
                hidden_state.data_ptr(),
                True,
            )
            cum_query_length += query_len

        # Run decodes.
        if len(decode_groups) > 0:
            session_ids_tensor = torch.tensor(session_ids,
                                              device=self.device,
                                              dtype=torch.int)
            self.scratch.decode(
                session_ids_tensor.data_ptr(),
                input_tokens_tensor.data_ptr(),
                batch_size,
                hidden_states.data_ptr(),
            )

        hidden_states = hidden_states.view(-1,
                                           self.model_config.get_hidden_size())
        # Scratch doesn't apply rms norm in its output, so we should do it
        # ourselves. Residual is set to None because it is already added
        # from Scratch output.
        hidden_states = self.model.norm(hidden_states, None)
        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

    def _execute_and_scratch_sample(
        self,
        prefill_groups: List[SequenceGroupMetadata],
        decode_groups: List[SequenceGroupMetadata],
        input_tokens: List[int],
        session_ids: List[int],
        parent_ids: List[int],
        query_lens: List[int],
    ) -> SamplerOutput:
        """Execute the Scratch kernels and its native sampler.
        
        Scratch sampler currently only supports greedy sampling. Logprobs
        reported from Scratch is inaccurate.

        Args:
            prefill_groups: A list of sequence group metadata for prefill
            decode_groups: A list of sequence group metadata for decode
            input_tokens: (num_batched_tokens) input tokens.
            session_ids: A list of session ids.
            parent_ids: A list of sequence ids.
            query_lens: (num_seqs). A length of input tokens per sequence.
                Used to find the length of input queries from 1D
                `input_tokens`.

        Returns:
            SamplerOutput for a given batch of requests.
        """
        batch_size = len(session_ids)
        tokens_out = torch.empty(batch_size,
                                 device=self.device,
                                 dtype=torch.int)
        input_tokens_tensor = torch.tensor(input_tokens,
                                           device=self.device,
                                           dtype=torch.int)
        cum_query_length = 0
        for i, session_id in enumerate(session_ids):
            query_len = query_lens[i]
            # Find relevant tensor from 1D input token tensors.
            prefill_req_tensor = input_tokens_tensor[
                cum_query_length:cum_query_length + query_len]
            self.scratch.prefill_sampled(session_id,
                                         prefill_req_tensor.data_ptr(),
                                         prefill_req_tensor.shape[0],
                                         tokens_out[i].data_ptr())
            cum_query_length += query_len

        if len(decode_groups) > 0:
            session_ids_tensor = torch.tensor(session_ids,
                                              device=self.device,
                                              dtype=torch.int)
            self.scratch.decode_sampled(
                session_ids_tensor.data_ptr(),
                input_tokens_tensor.data_ptr(),
                batch_size,
                tokens_out.data_ptr(),
            )

        result_tokens = tokens_out.tolist()
        outputs = []
        for result_token, parent_id in zip(result_tokens, parent_ids):
            outputs.append(
                CompletionSequenceGroupOutput(
                    samples=[
                        SequenceOutput(
                            parent_id,
                            result_token,
                            # NOTE: This value is invalid.
                            {result_token: Logprob(logprob=0.5)},
                        )
                    ],
                    prompt_logprobs=None,
                ))
        return SamplerOutput(outputs=outputs)

    @torch.inference_mode()
    def profile_run(self) -> None:
        # TODO(sang): Run it once Scratch uses vllm kv caches.
        return

    def remove_all_loras(self):
        raise NotImplementedError

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
