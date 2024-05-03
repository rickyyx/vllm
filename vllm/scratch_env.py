import os

SCRATCH_ENV_VAR = "ANYSCALE_USE_SCRATCH_LLM"
USE_SCRATCH = bool(os.getenv(SCRATCH_ENV_VAR, False))

if USE_SCRATCH:
    try:
        from vllm.scratch import ScratchAPI
    except ImportError:
        raise AssertionError(
            "Scratch API hasn't been built with vLLM properly. "
            "See https://docs.google.com/document/d/1O9VIfnhYai-gJ1TLlP-3SQ4wH5LqxafxYeEHmEIPD7Q/edit#heading=h.1j3ik15fr6mh") # noqa