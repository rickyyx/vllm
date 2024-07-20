#!/bin/bash

set -euo pipefail

# Check if the temporary directory was provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Provide a build directory as an argument."
    exit 1
fi

# Read the temporary directory from the command line argument
TMP_DIR="$1"
SCRATCH_DIR=${TMP_DIR}/scratchllm
# Remove the directory if already exists.
rm -rf "${SCRATCH_DIR}"

echo "Install ScratchLLM to ${SCRATCH_DIR}"
COMMIT="f86514c205c05e81717ae3357f80fc3f85326e34"
SCRATCH_REPO="scratchllm"
# Anyscale-legacy-work account
URI="s3://anyscale-test/scratch-repo-archive/${COMMIT}/${SCRATCH_REPO}.zip"
aws s3 cp "${URI}" "${TMP_DIR}"
pushd "${TMP_DIR}"
echo "unzip ${SCRATCH_REPO}.zip"
unzip "${SCRATCH_REPO}.zip"
rm "${SCRATCH_REPO}.zip"
popd

pushd "${SCRATCH_DIR}"
echo "Build glog"
wget https://github.com/google/glog/archive/refs/tags/v0.7.1.tar.gz
tar -xzvf v0.7.1.tar.gz
rm v0.7.1.tar.gz
pushd glog-0.7.1
cmake -S . -B build -G "Unix Makefiles"
cmake --build build
sudo cmake --build build --target install
popd

echo "Build scratchllm"
# used for pybind.
ls
bash setup_pybind.sh

# Build so files for A10.
make m=ll27b h=a10g t=f16 b=fullopt s=1 scratch_runner
make m=ll38b h=a10g t=f16 b=fullopt s=1 scratch_runner
# Build so files for H100.
make m=ll27b h=h100 t=f16 b=fullopt s=4 scratch_runner
make m=ll38b h=h100 t=f16 b=fullopt s=4 scratch_runner
popd
