from typing import List, Optional, Set, Hashable
import time

import torch
import torch.nn as nn
import boto3
from tqdm import tqdm
from pathlib import Path

from vllm.config import (
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VisionLanguageConfig,
    CacheConfig,
)
from pathlib import Path
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.sequence import (
    SamplerOutput,
    SequenceGroupMetadata,
    SequenceOutput,
    CompletionSequenceGroupOutput,
)
from vllm.model_executor.model_loader import get_model
from vllm.utils import CudaMemoryProfiler
from vllm.sequence import Logprob
from vllm.model_executor import SamplingMetadata
from vllm.utils import is_pin_memory_available
from vllm.utils import LRUCache

logger = init_logger(__name__)

LLAMA_7B_VOCAB_SIZE = 32000

from vllm.scratch import ScratchAPI
from vllm.scratch_env import (SCRATCH_TMP_DIR, SCRATCH_WEIGHTS_PREFIX,
                              SCRATCH_WEIGHTS_BUCKET_NAME)

# SANG-TODO WORKS?
MODEL_PARAMS_PATH = "/home/ray/default/weights"


class ScratchSession:

    def __init__(self, scratch_session_id: int):
        self.scratch_session_id = scratch_session_id


class ScratchLRUCache(LRUCache[ScratchSession]):

    def __init__(self, capacity: int, scratch_api: ScratchAPI):
        self._scratch_api = scratch_api
        super().__init__(capacity)

    def _on_remove(self, key: Hashable, value: ScratchSession):
        # Currently, key and values are both int session id.
        self._scratch_api.delete_session(value.scratch_session_id)


class ScratchSessionManager:
    """A class that manages multile scratch sessions.

    Stale sessions are currently automatically deleted. IT IS A HACK.
    The current implementation is kind of a hack because vLLM model runner
    cannot know which sequence groups are finished/preempted. We use LRUCache
    to track the max_num_seqs * 2 session and clean up session that are least
    recently used. It is working becasue vLLM scheduler prioritizes running
    decode all the time. It may break when preemption of swapping happens.
    Currently, we are disabling these features when ScratchLLM is used. This
    can be solved once we pipeline the finished/preempted sequence group
    information to model runner in a few weeks.
    """

    def __init__(self, scratch_api: ScratchAPI, max_num_seqs: int):
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


class ScratchModelRunner:

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
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.cache_config = cache_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.lora_manager = None
        self.model_config.enforce_eager = True
        self.vision_language_config = vision_language_config
        self.kv_cache_dtype = kv_cache_dtype
        self.pin_memory = is_pin_memory_available()

        # Lazily initialized.
        self.scratch: ScratchAPI
        # Scratch only returns embedding. We need to multiply it to lm_head
        # to get the final logits, and that happens in vLLM. In order to
        # do that, we create a torch module with lm_head weights loaded.
        self.model: nn.Module
        self._scratch_session_manager: ScratchSessionManager
        self._verify_scratch_config()

    def _verify_scratch_config(self):
        assert self.is_driver_worker, ("TP > 1 not supported.")
        assert self.model_config.dtype == torch.half, (
            "Only half type is allowed.")
        assert self.scheduler_config.chunked_prefill_enabled is False, (
            "Chunked prefill not supported")
        assert self.sliding_window is None, ("Sliding window not supported")
        assert self.vision_language_config is None, (
            "Vision model not supported")
        assert self.vision_language_config is None, (
            "Vision model not supported")
        assert self.kv_cache_dtype == "auto", (
            "Currently, Scratch doesn't use kv cache.")
        assert "llama-2" in self.model_config.model.lower(), (
            "Only Llama 7B is supported.")
        assert self.lora_manager is None, ("lora is not supported.")
        assert self.model_config.enforce_eager is True, (
            "cuda graph is not needed for Scratch.")
        # SANG-TODO page size should be 32.
        assert self.cache_config.block_size == 32, (
            "Scratch only supports page size of 32.")

    def load_model(self) -> None:
        assert self.load_config.download_dir is None
        tmp_dir = Path(SCRATCH_TMP_DIR)
        tmp_dir.mkdir(exist_ok=True)
        weights_dir = tmp_dir / "parameters"
        weights_dir.mkdir(exist_ok=True)
        # TODO(sang): Need to obtain this programmatically.
        download_dir = weights_dir / "ll27b-s1-cuda-f16-fullopt"
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
                vision_language_config=self.vision_language_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                cache_config=self.cache_config,
            )
            self.scratch = ScratchAPI(str(weights_dir.absolute()))
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
        assert len(dirs) == 1

        # NOTE(sang): Versioning is not supported now. We assume the
        # weights are always the same.
        # NOTE: Threadpool doesn't really improve performance.
        # Maybe it is rate limited.
        for file in tqdm(
                files, desc=f"Downloading scratch weights to {target_dir}"):
            dest = Path(target_dir) / Path(file).name
            if not dest.exists():
                s3_client.download_file(bucket, file, str(dest.absolute()))

    def set_block_size(self, block_size: int) -> None:
        # It will be relevant later.
        self.block_size = block_size

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        # KV cache is unused.
        input_tokens: List[int] = []
        parent_ids = []
        session_ids = []
        prefill_groups = []
        decode_groups = []
        query_lens = []
        seq_lens = []
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
                    # TODO(sang): Hack. Remove it.
                    input_tokens.append(data.prompt_token_ids)
                    query_lens.append(seq_data[parent_id].get_prompt_len())
                    seq_lens.append(seq_data[parent_id].get_prompt_len())
                else:
                    input_tokens.append(data.get_last_token_id())
                    query_lens.append(1)
                    seq_lens.append(seq_data[parent_id].get_len())

        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     seq_lens, query_lens,
                                                     self.device,
                                                     self.pin_memory)
        return self._execute_and_vllm_sample(prefill_groups, decode_groups,
                                             input_tokens, session_ids,
                                             parent_ids, sampling_metadata)
        # return self._execute_and_scratch_sample(
        #     prefill_groups, decode_groups, input_tokens, session_ids, parent_ids)

    def _execute_and_vllm_sample(
            self,
            prefill_groups: List[SequenceGroupMetadata],
            decode_groups: List[SequenceGroupMetadata],
            # It is 2D query if it is prefill, else decode.
            input_tokens: List[int],
            session_ids: List[int],
            parent_ids: List[int],
            sampling_metadata: SamplingMetadata):
        if len(prefill_groups) > 0:
            assert len(decode_groups) == 0
        if len(decode_groups) > 0:
            assert len(prefill_groups) == 0

        batch_size = len(session_ids)
        # TODO(ricky): This is not right when all_embeddings=True. 
        if len(prefill_groups) > 0:
            total_input_count = sum(
                [len(ins) for ins in input_tokens])
            hidden_states = torch.zeros(
                total_input_count *
                self.model_config.get_hidden_size(),
                device="cuda",
                dtype=torch.half)
        else:
            hidden_states = torch.zeros(self.model_config.get_hidden_size() *
                                        batch_size,
                                        device="cuda",
                                        dtype=torch.half)

        s = time.time()
        # Run prefills. Scratch currently doesn't support batch prefills, so we should
        # iterate one by one.
        for i, (session_id, prefill_group) in enumerate(zip(session_ids, prefill_groups)):
            input_tokens_tensor = torch.tensor(input_tokens[i],
                                               device="cuda",
                                               dtype=torch.int)
            print(f"SANG-TODO {input_tokens_tensor=}")
            assert input_tokens_tensor.is_contiguous()
            # print(f"SANG-TODO {input_tokens_tensor.shape=}")

            len_prefix_before_this = sum(
                len(ins) for ins in input_tokens[:i])
            # print(f"SANG-TODO {len_prefix_before_this=}")
            hidden_states_start_index = len_prefix_before_this * self.model_config.get_hidden_size()
            hidden_states_end_index = (len_prefix_before_this + len(input_tokens[i])) * self.model_config.get_hidden_size()
            # print(f"SANG-TODO {hidden_states_start_index=} {hidden_states_end_index=}")
            # print(f"SANG-TODO {hidden_states.shape=}")
            print(f"SANG-TODO {hidden_states[hidden_states_start_index: hidden_states_end_index].shape=}")
            assert hidden_states[hidden_states_start_index: hidden_states_end_index].is_contiguous()
            self.scratch.prefill(
                session_id,
                input_tokens_tensor.data_ptr(),
                input_tokens_tensor.shape[0],
                hidden_states[hidden_states_start_index: hidden_states_end_index].data_ptr(),
                True,
            )

        # Run decodes.
        if len(decode_groups) > 0:
            input_tokens_tensor = torch.tensor(input_tokens,
                                               device="cuda",
                                               dtype=torch.int)
            session_ids_tensor = torch.tensor(session_ids,
                                              device="cuda",
                                              dtype=torch.int)
            self.scratch.decode(
                session_ids_tensor.data_ptr(),
                input_tokens_tensor.data_ptr(),
                batch_size,
                hidden_states.data_ptr(),
            )

        print(
            f"SANG-TODO forward takes {(time.time() - s)* 1000} ms. Batch size: {len(session_ids)=} is_prefill: {len(prefill_groups) > 0}"
        )
        # print(hidden_states)
        # print(f"SANG-TODO {hidden_states.shape=}")
        # Post process Scratch embeddings.
        # RICKY-QQ: 
        # so for prefill, with 2D tensor below, we actually treating multiple prefills as a single batch, as we do not have the batch dimension in the tensor.
        # but for decode, we treat each decode as a single batch, as we have the batch dimension in the tensor.
        # is this expected? 
        hidden_states = hidden_states.view(-1,
                                           self.model_config.get_hidden_size())
        if len(prefill_groups) > 0:
            print(f"SANG-TODO before norm {hidden_states=}")
            print(f"SANG-TODO {hidden_states.shape=}")
        # Scratch doesn't apply rms norm in its output, so we should do it ourselves.
        # Residual is set to None because it is already added from Scratch output.
        hidden_states = self.model.norm(hidden_states, None)
        if len(prefill_groups) > 0:
            print(f"SANG-TODO norm weights: {self.model.norm.weight=}")
            print(f"SANG-TODO {hidden_states.shape=}")
            print(f"SANG-TODO after norm {hidden_states=}")
        # print(f"{hidden_states.shape=}")

        # SANG-TODO remove it. Hack. It will work once scrath returns embedding of all tokens correctly.
        # if len(prefill_groups) > 0:
        #     sampling_metadata.selected_token_indices = torch.tensor(
        #         [len(ins) - 1 for ins in input_tokens], device="cuda", dtype=torch.int)
        # else:
        #     sampling_metadata.selected_token_indices = torch.tensor(
        #         [1 for _ in range(batch_size)], device="cuda", dtype=torch.int)
        
        # print(f"{sampling_metadata.selected_token_indices=}")
        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        if len(prefill_groups) > 0:
            print(
                f"SANG-TODO prefill takes {(time.time() - s)* 1000} ms. Batch size: {len(session_ids)=}"
            )
        else:
            print(
                f"SANG-TODO decode takes {(time.time() - s)* 1000} ms. Batch size: {len(session_ids)=}"
            )
        # print(output)
        return output

    def _execute_and_scratch_sample(
            self,
            prefill_groups: List[SequenceGroupMetadata],
            decode_groups: List[SequenceGroupMetadata],
            input_tokens: List[int],
            session_ids: List[int],
            parent_ids: List[int]):
        batch_size = len(session_ids)
        tokens_out = torch.zeros(batch_size, device="cuda", dtype=torch.int)
        for i, (session_id, prefill_group) in enumerate(zip(session_ids, prefill_groups)):
            input_tokens_tensor = torch.tensor(input_tokens[i],
                                               device="cuda",
                                               dtype=torch.int)
            self.scratch.prefill_sampled(session_id,
                                         input_tokens_tensor.data_ptr(),
                                         input_tokens_tensor.shape[0],
                                         tokens_out[i].data_ptr())

        if len(decode_groups) > 0:
            input_tokens_tensor = torch.tensor(input_tokens,
                                               device="cuda",
                                               dtype=torch.int)
            session_ids_tensor = torch.tensor(session_ids,
                                              device="cuda",
                                              dtype=torch.int)
            self.scratch.decode_sampled(
                session_ids_tensor.data_ptr(),
                input_tokens_tensor.data_ptr(),
                batch_size,
                tokens_out.data_ptr(),
            )
        print(f"SANG-TODO token: {tokens_out}")

        result_tokens = tokens_out.tolist()
        outputs = []
        for result_token, parent_id in zip(result_tokens, parent_ids):
            outputs.append(
                CompletionSequenceGroupOutput(
                    samples=[
                        SequenceOutput(
                            parent_id,
                            result_token,
                            # This value is invalid.
                            {result_token: Logprob(logprob=0.5)},
                        )
                    ],
                    prompt_logprobs=None,
                )
            )
        output = SamplerOutput(outputs=outputs)
        print(output)
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # TODO(sang): Run profile run.
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
