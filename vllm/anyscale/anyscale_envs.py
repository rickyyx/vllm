import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    ENABLE_JSON_MODE: bool
    LOGIT_PROCESSOR_CLS: Optional[str]
    RECREATE_FAILED_ACTORS: bool = False
    DELAY_BETWEEN_ACTOR_RESTARTS_S: float = 0.0
    MAX_RESTARTS: int = -1
    USE_V2: bool = False
    NUM_PROCESSOR_WORKERS: int = 8
    USE_SCRATCH: bool = False
    USE_SCRATCH_SAMPLE: bool = False

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: Dict[str, Callable[[], Any]] = {
    # -- Json Mode Manager config --
    "ENABLE_JSON_MODE":
    lambda: bool(int(os.getenv("ANYSCALE_VLLM_ENABLE_JSON_MODE", False))),
    # The class to use for the logit processor.
    # If str, it should be the importable qualified name of the class.
    # e.g. "vllm.logit_processor.JSONModeLogitsProcessor".
    "LOGIT_PROCESSOR_CLS":
    lambda: os.getenv("ANYSCALE_VLLM_LOGIT_PROCESSOR_CLS", None),
    # Whether to recreate the actor if it fails.
    "RECREATE_FAILED_ACTORS":
    lambda: bool(int(os.getenv("ANYSCALE_VLLM_RECREATE_FAILED_ACTORS", False))
                 ),
    # The delay between actor restarts.
    "DELAY_BETWEEN_ACTOR_RESTARTS_S":
    lambda: float(
        os.getenv("ANYSCALE_VLLM_DELAY_BETWEEN_ACTOR_RESTARTS_S", 0.0)),
    # The maximum number of restarts before the actor is left dead.
    "MAX_RESTARTS":
    lambda: int(os.getenv("ANYSCALE_VLLM_MAX_RESTARTS", -1)),
    # Whether to use json mode v2.
    "USE_V2":
    lambda: bool(int(os.getenv("ANYSCALE_VLLM_USE_V2", False))),
    # The number of json mode manager processor workers.
    "NUM_PROCESSOR_WORKERS":
    lambda: int(os.getenv("ANYSCALE_VLLM_NUM_PROCESSOR_WORKERS", 8)),
    # If set, ScratchLLM is used.
    "USE_SCRATCH":
    lambda: bool(int(os.getenv("ANYSCALE_VLLM_USE_SCRATCH_LLM", False))),
    # If set, use ScratchLLM's native sampler. It currently only supports
    # greedy sampling without any logprob. It is just used for correctness
    # checks.
    "USE_SCRATCH_SAMPLE":
    lambda: bool(int(os.getenv("ANYSCALE_VLLM_USE_SCRATCH_SAMPLE", False))),
    # Used to pass HF_MODEL_ID to scratch. vLLM doesn't guarantee to pass
    # the correct hf_model_id, so we rely on this env var.
    "SCRATCH_HF_MODEL_ID":
    lambda: os.getenv("ANYSCALE_VLLM_SCRATCH_HF_MODEL_ID", None),
}

# end-env-vars-definition


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())