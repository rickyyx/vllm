#!/bin/bash

set -eo pipefail

# Check if the temporary directory was provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Provide a build directory as an argument."
    exit 1
fi

# Read the temporary directory from the command line argument
TMP_DIR="$1"
SCRATCH_DIR=${TMP_DIR}/scratchllm

echo "Install ScratchLLM to ${SCRATCH_DIR}"
rm -rf ${SCRATCH_DIR}
git clone git@github.com:anyscale/scratchllm.git ${SCRATCH_DIR}
pushd ${SCRATCH_DIR}

git checkout a10-deployment

echo "Build glog"
git clone https://github.com/google/glog.git
pushd glog
cmake -S . -B build -G "Unix Makefiles"
cmake --build build
sudo cmake --build build --target install
popd

echo "Build scratchllm"
# used for pybind.
chmod 700 setup_pybind.sh
bash setup_pybind.sh

# TODO(sang): Support custom flags.
# SANG-TODO H100
# make m=ll38b h=cuda t=f16 b=fullopt s=4 scratch_runner
# SANG-TODO A10
make m=ll38b h=cuda t=f16 b=fullopt s=1 scratch_runner
popd
