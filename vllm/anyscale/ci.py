# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.

import argparse
import glob
import json
import logging
import os
import time
from typing import Any, Dict

import boto3
import torch
from botocore.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("-p",
                "--path",
                required=True,
                help="path to the results json file")
ap.add_argument(
    "--mode",
    required=False,
    default="vllm",
    choices=["vllm", "llmval"],
    help="weather results format in llvmal or vllm.",
)
ap.add_argument("--commit", required=False, default="", help="commit hash")
ap.add_argument("--branch", required=False, default="", help="branch name")
ap.add_argument("--buildkite-url",
                required=False,
                default="",
                help="buildkite url")
ap.add_argument("--tp",
                required=False,
                default=1,
                type=int,
                help="tensor parallelism")
ap.add_argument("--name",
                required=False,
                default="manual",
                type=str,
                help="test-name")
ap.add_argument(
    "--dry-run",
    required=False,
    action="store_true",
    default=False,
    help="if upload should just be print",
)

logger = logging.getLogger(__name__)


def get_gpu_info() -> (str, int):
    """
    Get the GPU type and count on the node.
    """

    gpu_type = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    return gpu_type, gpu_count


def _report_result(table: str, data: Dict[str, Any], upload: bool = True):
    """
    Report benchmark results to DB.

    This uploads result to a firehose stream, consumed by a databricks job
    that reads data from the destination S3 bucket and writes to a delta lake.

    Args:
        result: A dictionary containing benchmark results.

    """
    try:
        result_json = {
            "_table": table,
            "report_ts": int(time.time()),
            **data,
        }
        if not upload:
            print("Not uploading: ")
            print(result_json)
            return

        logger.info("Persisting result to the databricks delta lake...")
        firehose = boto3.client("firehose",
                                config=Config(region_name="us-west-2"))
        firehose.put_record(
            DeliveryStreamName="vllm-ci-result",
            Record={"Data": json.dumps(result_json)},
        )
    except Exception as e:
        print("Failed to persist result to the databricks delta lake")
        print(e)
    else:
        print("Result has been persisted to the databricks delta lake")


def report_result_vllm(
    test_name: str,
    result: Dict[str, Any],
    ci_url: str,
    commit: str = "",
    branch: str = "",
    upload: bool = True,
):
    """
    Report benchmark results to DB.

    This uploads result to a firehose stream, consumed by a databricks job
    that reads data from the destination S3 bucket and writes to a delta lake.

    Args:
        result: A dictionary containing benchmark results.

    """

    data = {
        "status": "passed",
        "name": test_name,
        "commit": commit,
        "branch": branch,
        "buildkite_url": ci_url,
        "results": result,
    }
    _report_result(table="release_test_result", data=data, upload=upload)


def report_result_llmval(
    data: Dict,
    name: str,
    ci_url: str,
    commit: str = "",
    branch: str = "",
    upload: bool = True,
):
    data.update({
        "name": name,
        "buildkite_url": ci_url,
        "commit": commit,
        "branch": branch
    })
    _report_result(table="llmval_perf_results", data=data, upload=upload)


def parse_result(data: dict, tp: int) -> dict:
    """
    Parse the result from the json file.
    """

    metric = data.get("metadata", {})
    gpu_type, gpu_count = get_gpu_info()
    return {
        "model_name":
        metric.get("model", "").lower(),
        "input_len":
        metric.get("num_input_tokens", 0),
        "output_len":
        metric.get("num_output_tokens", 0),
        "num_users":
        metric.get("num_users", 0),
        "tp":
        tp,
        "gpu_type":
        gpu_type,
        "gpu_count":
        gpu_count,
        "tokens_per_s":
        metric.get("overall_throughput_token_per_s", 0),
        "perf_metrics": [
            {
                "perf_metric_name":
                "tokens_per_s",
                "perf_metric_value":
                metric.get("overall_throughput_token_per_s", 0),
                "perf_metric_type":
                "THROUGHPUT",
            },
            {
                "perf_metric_name": "p50_token_latency",
                "perf_metric_value": metric.get("token_lat_s_p50", 0),
                "perf_metric_type": "LATENCY",
            },
            {
                "perf_metric_name": "p90_token_latency",
                "perf_metric_value": metric.get("token_lat_s_p90", 0),
                "perf_metric_type": "LATENCY",
            },
            {
                "perf_metric_name": "p99_token_latency",
                "perf_metric_value": metric.get("token_lat_s_p99", 0),
                "perf_metric_type": "LATENCY",
            },
            {
                "perf_metric_name": "p50_e2e_latency",
                "perf_metric_value":
                metric.get("user_total_request_time_s_p50", 0),
                "perf_metric_type": "LATENCY",
            },
            {
                "perf_metric_name": "p90_e2e_latency",
                "perf_metric_value":
                metric.get("user_total_request_time_s_p90", 0),
                "perf_metric_type": "LATENCY",
            },
            {
                "perf_metric_name": "p99_e2e_latency",
                "perf_metric_value":
                metric.get("user_total_request_time_s_p99", 0),
                "perf_metric_type": "LATENCY",
            },
        ],
    }


def cli_upload(
    path: str,
    mode: str,
    commit: str,
    branch: str,
    buildkite_url: str,
    tp: int,
    name: str,
    dry_run: bool,
) -> None:
    if mode == "vllm":
        # Read from the json file
        with open(path) as f:
            data = json.load(f)

        # For legacy perf metrics reporting.
        result = parse_result(data, tp=tp)

        report_result_vllm(
            test_name=data.get("name", "NA"),
            result=result,
            ci_url=buildkite_url,
            commit=commit,
            branch=branch,
            upload=not dry_run,
        )
    elif mode == "llmval":
        # Use the glob module to get all files matching the pattern
        files = glob.glob(path)

        # Exclude files with the following patterns
        # 1. *_raw.json
        files = [file for file in files if not file.endswith("_raw.json")]

        if not files:
            print(f"No files found matching pattern: {path}")
            return

        # Simulated upload process
        for file in files:
            if os.path.isfile(file):  # Ensure it's a file and not a directory
                print(f"Uploading {file}...")
                # Read the json file
                with open(file) as f:
                    data = json.load(f)
                    data.update({"tp": tp})

                    report_result_llmval(
                        data,
                        name,
                        ci_url=buildkite_url,
                        commit=commit,
                        branch=branch,
                        upload=not dry_run,
                    )
            else:
                print(f"{file} is not a valid file.")
        print("Upload completed!")


if __name__ == "__main__":
    args = vars(ap.parse_args())
    cli_upload(**args)
