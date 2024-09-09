from pathlib import Path

SCRATCH_EXECUTABLE_PATH_ENV_VAR = "SCRATCH_EXECUTABLE_PATH"
# We should remove this, this is needed because weights are the
# same for all builds types.
SCRATCH_BUILD_TYPE = "fullopt"
ROOT_DIRECTORY = str(Path(__file__).parents[2])
SCRATCH_WEIGHTS_BUCKET_NAME = "large-dl-models-mirror"
SCRATCH_WEIGHTS_PATH_PREFIX = "scratchllm"
# This points to the scratchllm commit for the current compatible weights.
SCRATCH_WEIGHTS_VERSION_COMMIT = "f3259e0129418dc87e848c0d75ff4d44f8b992f7"
SCRATCH_TMP_DIR = "/tmp/scratch/"
