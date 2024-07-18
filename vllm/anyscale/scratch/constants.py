from pathlib import Path

SCRATCH_EXECUTABLE_PATH_ENV_VAR = "SCRATCH_EXECUTABLE_PATH"
# TODO(sang): H100
# SCRATCH_BUILD_PREFIX = "ll38b-s4-cuda-f16" # CHANGE THIS FOR DIFFERENT MODELS
# TODO(sang): A10
SCRATCH_BUILD_PREFIX_LLAMA_2 = "ll27b-s1-a10g-f16"
SCRATCH_BUILD_PREFIX_LLAMA_3 = "ll38b-s1-a10g-f16"
SCRATCH_BUILD_PREFIX_LLAMA_3_INST = "ll38b-inst-s1-a10g-f16"
# We should remove this, this is needed because weights are the
# same for all builds types.
SCRATCH_BUILD_TYPE = "fullopt"
ROOT_DIRECTORY = str(Path(__file__).parents[2])
SCRATCH_WEIGHTS_BUCKET_NAME = "scratch-working-dirs"
SCRATCH_TMP_DIR = "/tmp/scratch/"
SCRATCH_WEIGHTS_PATH = "/tmp/scratch/"
