#!/usr/bin/env bash
set -euxo pipefail

# install anyguide
ANYSCALE_BUCKET="s3://python-wheel-cache"
ANYSCALE_PROJECT="anyscale/anyguide"
ANYSCALE_COMMIT="90edfac0a2a648fa652bfb78669dea76dfa77241"
ANYSCALE_WHEEL="anyguide-0.1-cp39-cp39-linux_x86_64.whl"

echo "--- :s3: Pulling anyguide wheels ..."
aws s3 cp "${ANYSCALE_BUCKET}/${ANYSCALE_PROJECT}/${ANYSCALE_COMMIT}/${ANYSCALE_WHEEL}" "$ANYSCALE_PROJECT/$ANYSCALE_WHEEL"
pip install --no-deps "$ANYSCALE_PROJECT/$ANYSCALE_WHEEL"
pip install cachetools