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

chmod 700 .buildkite/ci/install_scratch_dependencies.sh
bash .buildkite/ci/install_scratch_dependencies.sh

echo "~~~ :python: Building wheel for ${VLLM_PROJECT}@${GIT_COMMIT}"
# Build scratch together.
BUILD_BAZEL=1 python setup.py bdist_wheel

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
