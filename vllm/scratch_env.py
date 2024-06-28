import os

SCRATCH_ENV_VAR = "ANYSCALE_USE_SCRATCH_LLM"
USE_SCRATCH = bool(int(os.getenv(SCRATCH_ENV_VAR, False)))
SCRATCH_EXECUTABLE_PATH_ENV_VAR = "SCRATCH_EXECUTABLE_PATH"
# SANG-TODO H100
# SCRATCH_BUILD_PREFIX = "ll38b-s4-cuda-f16" # CHANGE THIS FOR DIFFERNT MODELS
# SANG-TODO A10
SCRATCH_BUILD_PREFIX = "ll38b-s1-cuda-f16" # CHANGE THIS FOR DIFFERNT MODELS
SCRATCH_BUILD_TYPE = "fullopt" # We should remove this, this is needed because weights are the same for all builds types.
# Hack.
current_directory = os.path.dirname(os.path.abspath(__file__))
SCRATCH_EXECUTABLE_PATH =os.getenv(SCRATCH_EXECUTABLE_PATH_ENV_VAR, f"{current_directory}/scratch-{SCRATCH_BUILD_PREFIX}-{SCRATCH_BUILD_TYPE}.cpython-39-x86_64-linux-gnu.so")
SCRATCH_WEIGHTS_BUCKET_NAME = "scratch-working-dirs"
SCRATCH_WEIGHTS_PREFIX = f"staging_weights/{SCRATCH_BUILD_PREFIX}/"
SCRATCH_WEIGHTS_URI = f"s3://{SCRATCH_WEIGHTS_BUCKET_NAME}/{SCRATCH_WEIGHTS_PREFIX}"
SCRATCH_TMP_DIR = "/tmp/scratch/"
SCRATCH_WEIGHTS_PATH = "/tmp/scratch/"
