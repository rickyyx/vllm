from typing import List, Optional, Set
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

logger = init_logger(__name__)

LLAMA_7B_VOCAB_SIZE = 32000

from vllm.scratch import ScratchAPI
from vllm.scratch_env import (SCRATCH_TMP_DIR, SCRATCH_WEIGHTS_PREFIX,
                              SCRATCH_WEIGHTS_BUCKET_NAME)

# SANG-TODO WORKS?
MODEL_PARAMS_PATH = "/home/ray/default/weights"


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
        # session_id_str -> session_id
        self.session_ids = {}

        self._verify_scratch_config()

    def _verify_scratch_config(self):
        assert self.scheduler_config.max_num_seqs == 1, (
            "bsize > 1 is not supported.")
        assert self.is_driver_worker, ("TP > 1 not supported.")
        assert self.model_config.dtype == torch.half, ("Only half type is allowed.")
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

    def load_model(self) -> None:
        assert self.load_config.download_dir is None
        tmp_dir = Path(SCRATCH_TMP_DIR)
        tmp_dir.mkdir(exist_ok=True)
        weights_dir = tmp_dir / "parameters"
        weights_dir.mkdir(exist_ok=True)
        # TODO(sang): Need to obtain this programmatically.
        download_dir = weights_dir / "ll27b-cuda-f16-fullopt"
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
        for file in tqdm(files, desc=f"Downloading scratch weights to {target_dir}..."):
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
        assert len(seq_group_metadata_list) == 1, "Only bsize 1 is allowed."

        is_prefill = False
        session_id: Optional[int] = None
        parent_id = None

        query_lens = []
        seq_lens = []
        for seq_group_metadata in seq_group_metadata_list:
            is_prefill = seq_group_metadata.is_prompt
            seq_data = seq_group_metadata.seq_data
            # Scratch only supports a single sequence.
            assert len(seq_data) == 1
            for seq_id, data in seq_data.items():
                parent_id = seq_id
                if is_prefill:
                    input_tokens.extend(data.prompt_token_ids)
                else:
                    # pass
                    input_tokens.append(data.get_token_ids()[-1])
                    # input_tokens.append(data.get_token_ids())

            assert parent_id is not None
            session_id = int(seq_group_metadata.request_id)
            if is_prefill:
                query_lens.append(seq_data[parent_id].get_prompt_len())
                seq_lens.append(seq_data[parent_id].get_prompt_len())
            else:
                query_lens.append(1)
                seq_lens.append(seq_data[parent_id].get_len())

        assert session_id is not None
        assert parent_id is not None
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list, seq_lens, query_lens, self.device,
            self.pin_memory)

        if session_id not in self.session_ids:
            # SANG-TODO Remove the logic after supporting multi sessions.
            if len(self.session_ids) != 0:
                self.session_ids = {}
            self.session_ids[session_id] = self.scratch.new_session()
            print(self.session_ids)

        # print(f"SANG-TODO {input_tokens=}")
        return self._execute_and_vllm_sample(is_prefill, input_tokens, session_id, parent_id, sampling_metadata)
        # return self._execute_and_scratch_sample(is_prefill, input_tokens, session_id, parent_id)

    def _execute_and_vllm_sample(self, is_prefill: bool, input_tokens: List[int], session_id: int, parent_id: int, sampling_metadata: SamplingMetadata):
        input_tokens_tensor = torch.tensor(input_tokens, device="cuda", dtype=torch.int)
        batch_size = 1
        hidden_states = torch.zeros(
            self.model_config.get_hidden_size() * batch_size, device="cuda", dtype=torch.half)
        session_id = self.session_ids[session_id]

        s = time.time()
        if is_prefill:
            self.scratch.prefill(
                session_id,
                input_tokens_tensor.data_ptr(),
                input_tokens_tensor.shape[0],
                hidden_states.data_ptr()
            )
        else:
            session_ids_tensor = torch.tensor([session_id], device="cuda", dtype=torch.int)
            self.scratch.decode(
                session_ids_tensor.data_ptr(),
                input_tokens_tensor.data_ptr(),
                batch_size,
                hidden_states.data_ptr(),
            )
        print(f"SANG-TODO forward takes {(time.time() - s)* 1000} ms")

        # SANG-TODO remove it. Hack.
        sampling_metadata.selected_token_indices = torch.tensor([0], device="cuda", dtype=torch.int)

        logits = self.model.compute_logits(hidden_states.view(-1, self.model_config.get_hidden_size()), sampling_metadata)
        # if is_prefill:
        #     print("SANG-TODO hidden_states after prefill:")
        #     print(hidden_states)
        #     print("SANG-TODO logits after prefill:")
        #     print(logits)

        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        if is_prefill:
            print(f"SANG-TODO prefill takes {(time.time() - s)* 1000} ms")
        else:
            print(f"SANG-TODO decode takes {(time.time() - s)* 1000} ms")
        return output

    def _execute_and_scratch_sample(self, is_prefill: bool, input_tokens: List[int], session_id: object, parent_id: int):
        input_tokens_tensor = torch.tensor(input_tokens, device="cuda", dtype=torch.int)
        tokens_out = torch.zeros(1, device="cuda", dtype=torch.int)
        batch_size = 1
        session_id = self.session_ids[session_id]

        # prefill_sampled
        if is_prefill:
            s = time.time()
            self.scratch.prefill_sampled(
                session_id,
                input_tokens_tensor.data_ptr(),
                input_tokens_tensor.shape[0],
                tokens_out.data_ptr())
            print(f"SANG-TODO prefill takes {(time.time() - s)* 1000} ms")
        else:
            s = time.time()
            session_ids_tensor = torch.tensor([session_id], device="cuda", dtype=torch.int)
            self.scratch.decode_sampled(
                session_ids_tensor.data_ptr(),
                input_tokens_tensor.data_ptr(),
                batch_size,
                tokens_out.data_ptr(),
            )
            print(f"SANG-TODO decode takes {(time.time() - s)* 1000} ms")
        print(f"SANG-TODO token: {tokens_out}")

        result_token = tokens_out.tolist()[0]
        # Logprob/prompt logprob not supported. It should work once sampler
        # is supported.
        return SamplerOutput(outputs=[
            CompletionSequenceGroupOutput(
                samples=[
                    SequenceOutput(
                        parent_id,
                        result_token,
                        {result_token: Logprob(logprob=0.5)},  # logprob
                    )
                ],
                prompt_logprobs=None,
            )
        ])

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
