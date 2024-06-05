#!/bin/bash
set -euxo pipefail

S3_WHEEL_CACHE="s3://anyscale-test/wheels"
VLLM_PROJECT="anyscale/vllm"
VLLM_WHEEL="vllm-0.1.4-cp39-cp39-linux_x86_64.whl"
# We need to edit ldconfig directly as Triton AOT depends on that
# echo "/usr/local/cuda/lib64/stubs" | sudo tee /etc/ld.so.conf.d/001_cuda_stubs.conf
# sudo ldconfig

rm -rf dist
rm -rf build

# Download cmake
sudo apt-get update
sudo apt-get install -y cmake

echo "~~~ :python: Building wheel for ${VLLM_PROJECT}@${GIT_COMMIT}"
BUILD_BAZEL=1 python setup.py bdist_wheel

COMMIT_PATH="${S3_WHEEL_CACHE}/${VLLM_PROJECT}/${GIT_COMMIT}/${VLLM_WHEEL}"
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
