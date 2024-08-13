#!/usr/bin/env bash

set -euo pipefail
source .buildkite/ci/bash_util/timeout.sh

# run unit test
run_with_timeout $(( 60 * 60 )) bash -c "cd anyscale; bazel test s3/..."
run_with_timeout $(( 60 * 60 )) pytest -vs tests/anyscale/s3_tensor/test_s3_tensor_client.py
run_with_timeout $(( 60 * 60 )) pytest -vvs --forked tests/anyscale/s3_tensor/test_s3_e2e.py
