#!/bin/bash

set -euo pipefail
echo "Uploading pipeline..."
ls .buildkite || buildkite-agent annotate --style error 'Please merge upstream main branch for buildkite CI'
curl -sSfL https://github.com/mitsuhiko/minijinja/releases/latest/download/minijinja-cli-installer.sh | sh
source /var/lib/buildkite-agent/.cargo/env
cd .buildkite
python3 merge_test_pipeline.py
minijinja-cli template.j2 merged-test-pipeline.yaml > pipeline.yml
cat pipeline.yml
buildkite-agent pipeline upload pipeline.yml
exit 0
