#!/usr/bin/env bash

set -euo pipefail

source .buildkite/ci/bash_util/timeout.sh

run_with_timeout $(( 90 * 60 )) pytest -vs tests/kernels
