#!/usr/bin/env bash

set -exuo pipefail

# Required: S3_SECRET_PATH - s3://<bucket>/<path> to text file containing Anyscale token
# Required: IMAGE - Full path to Docker image to benchmark
# Required: JOB_NAME - Name for job in Anyscale
# Required: COMPUTE_CONFIG - Path to compute config
# Required: COMMAND - Command to run on Anyscale

export PATH=$PATH:/var/lib/buildkite-agent/.local/bin
export ARTIFACTS_DIR=result/
export S3_SECRET_PATH=s3://bk-premerge-first-jawfish-secrets/aviary-performance/anyscale_token.txt
# The path that where artifacts from our job may be uploaded to.
# aviary org id, aviary-prod cloud id
export ANYSCALE_S3_ARTIFACT_PATH=s3://anyscale-production-data-cld-ldm5ez4edlp7yh4yiakp2u294w/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/vllm-ci-artifacts/
export IMAGE="${IMAGE:-241673607239.dkr.ecr.us-west-2.amazonaws.com/aviary:vllm-ci}"
export JOB_NAME=vllm-$BUILDKITE_JOB_ID-$BUILDKITE_RETRY_COUNT-g5-2xlarge

DOWNLOAD_ARTIFACTS=false
while [ "$#" -gt 0 ]; do
    case "$1" in 
        --download-s3-artifacts)
            DOWNLOAD_ARTIFACTS=true
            shift
            ;;
        *)
    esac
done

_term() {
    echo "+++ :skull: termination signal caught - killing benchmark process"
    kill -INT "$1" ||:
    wait "$1"
}

echo "--- :python: Installing dependencies"
python3 -m pip install -U anyscale==0.24.9 typer awscli awscrt==0.19.17 "urllib3<2.0"
mkdir -pv "${ARTIFACTS_DIR}"

echo "--- :aws: Fetching Anyscale secret"
set +x # stop command debugging that could print the secret
aws s3 cp "$S3_SECRET_PATH" "anyscale_token.txt"
export ANYSCALE_CLI_TOKEN=$(cat anyscale_token.txt)
rm anyscale_token.txt
set -x # re-enable command debugging

echo "--- :anyscale: Check anyscale installed"
anyscale --version

echo "--- :anyscale: Running job on Anyscale"
python3 .buildkite/ci/run_anyscale_job.py \
    "${COMPUTE_CONFIG}" \
    .buildkite/ci/cluster-env-template.yaml \
    .buildkite/ci/anyscale-job-template.yaml \
    "$(pwd)" \
    "${IMAGE}" \
    "${BUILDKITE_COMMIT}" \
    "${BUILDKITE_BRANCH}" \
    "${BUILDKITE_BUILD_URL}" \
    "${BUILDKITE_JOB_ID}" \
    "${JOB_NAME}" \
    "${COMMAND}" \
    "${ARTIFACTS_DIR}/${JOB_NAME}.log" &

child=$!
trap '_term $child' TERM INT
wait "$child"

if $DOWNLOAD_ARTIFACTS; then
    # Download the S3 artifacts locally.
    aws s3 cp "$ANYSCALE_S3_ARTIFACT_PATH" "$ARTIFACTS_DIR" --recursive
    # Delete the S3 artifacts.
    aws s3 rm "$ANYSCALE_S3_ARTIFACT_PATH" --recursive

    # Zip all subdirectories in $ARTIFACTS_DIR
    for dir in "$ARTIFACTS_DIR"/*/; do
        base=$(basename "$dir")
        zip -r "$ARTIFACTS_DIR/$base.zip" "$dir"
    done
fi
