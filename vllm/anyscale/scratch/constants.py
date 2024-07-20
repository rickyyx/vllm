from pathlib import Path

SCRATCH_EXECUTABLE_PATH_ENV_VAR = "SCRATCH_EXECUTABLE_PATH"
# We should remove this, this is needed because weights are the
# same for all builds types.
SCRATCH_BUILD_TYPE = "fullopt"
ROOT_DIRECTORY = str(Path(__file__).parents[2])
SCRATCH_WEIGHTS_BUCKET_NAME = "scratch-working-dirs"
SCRATCH_TMP_DIR = "/tmp/scratch/"
SCRATCH_WEIGHTS_PATH = "/tmp/scratch/"
