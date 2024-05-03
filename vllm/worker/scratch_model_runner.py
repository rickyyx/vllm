from typing import List, Optional, Set

import torch

from vllm.config import (
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VisionLanguageConfig,
)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.sequence import (
    SamplerOutput,
    SequenceGroupMetadata,
    SequenceOutput,
    SequenceGroupOutput,
)
from vllm.utils import CudaMemoryProfiler
from vllm.sequence import Logprob

logger = init_logger(__name__)

LLAMA_7B_VOCAB_SIZE = 32000

import random

from vllm.scratch import ScratchAPI

# MODEL_PARAMS_PATH = "/home/ubuntu/data/parameters/ll27b-cuda-f16-fullopt"
MODEL_PARAMS_PATH = "/home/ray/default/weights"


class ScratchModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
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

        self.scratch = ScratchAPI()

        self._verify_scratch_config()

    def _verify_scratch_config(self):
        assert self.scheduler_config.max_num_seqs == 1, (
            "bsize > 1 is not supported.")
        assert self.is_driver_worker, ("TP > 1 not supported.")
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
        with CudaMemoryProfiler() as m:
            self.scratch.load_model(MODEL_PARAMS_PATH)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        # KV cache dtype/quantization is not supported.

    def set_block_size(self, block_size: int) -> None:
        # It will be relevant later.
        self.block_size = block_size

    # NOTE: Scratch doesn't use torch.
    # @torch.inference_mode()
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
        for seq_group_metadata in seq_group_metadata_list:
            seq_data = seq_group_metadata.seq_data
            # Scratch only supports a single sequence.
            assert len(seq_data) == 1
            prompt_token_ids = []
            for seq_id, data in seq_data.items():
                parent_id = seq_id
                prompt_token_ids = data.prompt_token_ids
            session_id = int(seq_group_metadata.request_id)
            is_prefill = seq_group_metadata.is_prompt

            if is_prefill:
                input_tokens.extend(prompt_token_ids)

        assert session_id is not None
        assert parent_id is not None

        if is_prefill:
            result_token = self.scratch.prefill(input_tokens, session_id)
        else:
            result_token = self.scratch.decode(session_id)

        # Logprob/prompt logprob not supported. It should work once sampler
        # is supported.
        return SamplerOutput(outputs=[
            SequenceGroupOutput(
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
