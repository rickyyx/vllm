import os

SCRATCH_ENV_VAR = "ANYSCALE_USE_SCRATCH_LLM"
USE_SCRATCH = bool(int(os.getenv(SCRATCH_ENV_VAR, False)))
SCRATCH_WEIGHTS_BUCKET_NAME = "scratch-working-dirs"
SCRATCH_WEIGHTS_PREFIX = "weights/llama-7b/ll27b-cuda-f16/"
SCRATCH_WEIGHTS_URI = f"s3://{SCRATCH_WEIGHTS_BUCKET_NAME}/{SCRATCH_WEIGHTS_PREFIX}"
SCRATCH_TMP_DIR = "/tmp/scratch/"
SCRATCH_WEIGHTS_PATH = "/tmp/scratch/"

if USE_SCRATCH:
    try:
        from vllm.scratch import ScratchAPI
    except ImportError:
        raise AssertionError(
            "Scratch API hasn't been built with vLLM properly. "
            "See https://docs.google.com/document/d/1O9VIfnhYai-gJ1TLlP-3SQ4wH5LqxafxYeEHmEIPD7Q/edit#heading=h.1j3ik15fr6mh"
        )  # noqa
