"""Verify compatibility of overloaded class for scratchLLM.
"""

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models import ModelRegistry, OriginalModelRegistry

from vllm.anyscale.anyscale_envs import USE_SCRATCH

MODELS = [
    # "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
]

assert USE_SCRATCH, ("ScratchLLM should be enabled to run a test. "
                     "Use ANYSCALE_VLLM_USE_SCRATCH_LLM=1 pytest -vs "
                     "tests/scratch/anyscale/test_basic_correctness.py")


def test_model_registry_duck_typed() -> None:
    # Get the set of methods for each class
    engine_config = EngineArgs(
        "meta-llama/Meta-Llama-3-8B").create_engine_config()

    def find_apis(cls):
        methods = set()
        for name in dir(cls):
            # Exclude magic methods.
            if name.startswith("__") and name.endswith("__"):
                continue
            # Exclude private methods.
            if name.startswith("_"):
                continue
            methods.add(name)
        return methods

    methods1 = find_apis(ModelRegistry)
    methods2 = find_apis(OriginalModelRegistry)
    assert methods1 == methods2

    arch = getattr(engine_config.model_config.hf_config, "architectures", [])

    with pytest.raises(NotImplementedError):
        ModelRegistry.register_model(arch, None)

    assert ModelRegistry.is_embedding_model(arch) is False
