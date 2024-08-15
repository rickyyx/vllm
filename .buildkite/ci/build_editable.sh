#!/usr/bin/env bash
set -euxo pipefail

# Required to build scratchLLM.
ssh-keyscan github.com >> ~/.ssh/known_hosts
bash .buildkite/ci/install_scratch_dependencies.sh

# install vllm
pip install -U pip
pip install --no-cache -e .

# install anyguide
bash .buildkite/ci/install_anyguide.sh
