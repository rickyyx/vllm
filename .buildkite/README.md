test-pipeline.yaml is the buildkite yaml for OSS vLLM. We should eventually have a way to convert this into
our buildkite config, but it is not available.

pipeline-perf-ci.yaml is not working yet. We should convert this to use llmbench.

The following files are anyscale proprietary.
- ci folder
- job_submit_buildkite.sh
- pipeline-perf-ci.yml (to run perf test)
- pipeline-vllm-ci.yml (to run CI build)

Everything else is from OSS vLLM, and they are not used.
