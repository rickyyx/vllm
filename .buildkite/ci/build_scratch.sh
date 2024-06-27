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

# TEMPORARY.
# git checkout <commit>

echo "Build glog"
git clone https://github.com/google/glog.git
pushd glog
cmake -S . -B build -G "Unix Makefiles"
cmake --build build
sudo cmake --build build --target install
popd

# echo "Build sentencepiece"
# git clone https://github.com/google/sentencepiece.git 
# pushd sentencepiece
# mkdir build
# cd build
# cmake ..
# make -j $(nproc)
# sudo make install
# sudo ldconfig -v
# popd
# 
# echo "Build tiktokencpp"
# git clone git@github.com:anyscale/tiktokencpp.git
# pushd tiktokencpp
# mkdir build
# cd build
# cmake ..
# make
# sudo make install
# popd

echo "Build scratchllm"
# used for pybind.
git submodule update --init --recursive
# TODO(sang): Support custom flags.
make m=ll38b h=cuda t=f16 b=fullopt s=4 scratch_runner
popd
