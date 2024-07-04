#!/bin/bash
set -euxo pipefail

S3_WHEEL_CACHE="s3://anyscale-test/wheels"
VLLM_PROJECT="anyscale/vllm"
# We need to edit ldconfig directly as Triton AOT depends on that
# echo "/usr/local/cuda/lib64/stubs" | sudo tee /etc/ld.so.conf.d/001_cuda_stubs.conf
# sudo ldconfig

rm -rf dist
rm -rf build

# Download the latest cmake. vLLM requires cmake 3.22. It is easy to download in ubuntu 22.04, but we are using ubuntu 20.04.
# https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
# sudo apt-get update
# sudo apt-get install -y cmake
sudo apt update && \
sudo apt install -y software-properties-common lsb-release && \
sudo apt clean all
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install kitware-archive-keyring
sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4
sudo apt update
sudo apt install -y cmake

# Do not compile with debug symbol to reduce wheel size
export CMAKE_BUILD_TYPE="Release"

python_executable=python$1
cuda_home=/usr/local/cuda-$2

# Update paths
PATH=${cuda_home}/bin:$PATH
LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

# Install requirements
$python_executable -m pip install wheel packaging
$python_executable -m pip install -r requirements-cuda.txt

# Limit the number of parallel jobs to avoid OOM
export MAX_JOBS=24
# Make sure punica is built for the release (for LoRA)
export VLLM_INSTALL_PUNICA_KERNELS=1
# Make sure release wheels are built for the following architectures
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"

# Build dependencies required for ScratchLLM.
chmod 700 .buildkite/ci/install_scratch_dependencies.sh && bash .buildkite/ci/install_scratch_dependencies.sh

echo "~~~ :python: Building wheel for ${VLLM_PROJECT}@${GIT_COMMIT}"
# NOTE(sang): Scratch .so is automatically included.
$python_executable setup.py bdist_wheel --dist-dir=dist


VLLM_WHEEL=$(basename $(ls dist/*.whl))
COMMIT_PATH="${S3_WHEEL_CACHE}/${VLLM_PROJECT}/${GIT_COMMIT}/${VLLM_WHEEL}"
echo "~~~ :python: ls"
ls

echo "~~~ : Uploading to ${COMMIT_PATH}"
aws s3 cp "dist/${VLLM_WHEEL}" "${COMMIT_PATH}"

if [[ -n "$GIT_BRANCH" ]]
then
    echo "GIT_BRANCH: ${GIT_BRANCH}"

    if [[ "$GIT_BRANCH" == "main" ]]
    then
        BRANCH_PATH="${S3_WHEEL_CACHE}/${VLLM_PROJECT}/latest/${VLLM_WHEEL}"
        echo "~~~ : Uploading to ${BRANCH_PATH}"
        aws s3 cp "dist/${VLLM_WHEEL}" "${BRANCH_PATH}"
    else
        echo "Updating latest branch will only apply for main branch."
    fi
else
    echo "Error: GIT_BRANCH is not set"
fi