#!/usr/bin/env bash
set -euxo pipefail

# install scratchllm
ANYSCALE_BUCKET="s3://python-wheel-cache"
ANYSCALE_PROJECT="anyscale/scratchllm"
ANYSCALE_COMMIT="73e1170c0d1bfe41f4d6ae4daabd226bd0fa6b06"
ANYSCALE_WHEEL="scratchllm-0.1.0-cp39-cp39-linux_x86_64.whl"

echo "--- :s3: Pulling scratchllm wheels ..."
aws s3 cp "${ANYSCALE_BUCKET}/${ANYSCALE_PROJECT}/${ANYSCALE_COMMIT}/${ANYSCALE_WHEEL}" "$ANYSCALE_PROJECT/$ANYSCALE_WHEEL"
pip install pybind11
pip install "$ANYSCALE_PROJECT/$ANYSCALE_WHEEL"
