import itertools
from typing import Dict, Optional


default_envs = {
    "USE_DENSE_MOE": "0",
}

cmd_template = """
USE_DENSE_MOE={use_dense_moe} python benchmarks/benchmark_latency.py \
--model=mistralai/Mixtral-8x7B-Instruct-v0.1 \
--download-dir /mnt/local_storage \
--input-len {input_len} \
--output-len {output_len} \
-tp {tp} \
--num-iters 15 --batch-size {bs}
"""


TPS = [
    8, 4
]

USE_DENSE_MOES = [
    "1", "0"
]

INPUT_LEN = [
    5
]

OUTPUT_LEN = [
    512 
]

BATCH_SIZE = [
    1, 2, 4, 8, 16
]


class CsvResultTable:
    def __init__(self, cols, output_file):
        self.cols = cols
        self.rows = []
        self.output_file = output_file

    def add_row(self, **kwargs):
        assert set(kwargs.keys()) == set(self.cols)
        self.rows.append(",".join([str(kwargs[k]) for k in self.cols]))

    def dump(self):
        with open(self.output_file, "w") as f:
            f.write(",".join(self.cols) + "\n")
            for row in self.rows:
                f.write(row + "\n")


def parse_out(out: str) -> Dict:
    import re
    match = re.search(r"Avg latency: (\d+\.\d+) seconds", out)
    if match is None:
        return None
    e2e_latency = float(match.group(1))

    match = re.search(r"Model Median: (\d+\.\d+)", out)
    if match is None:
        return {
            "latency": e2e_latency,
            "model_latency": -1 ,
        }
    model_latency = float(match.group(1))

    return {
        "latency": e2e_latency,
        "model_latency": model_latency
    }


def run_cmd(cmd: str) -> Optional[Dict]:
    import subprocess

    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    out, err = p.communicate()

    if p.returncode != 0:
        print(f"Error: {err}")
        return None
    print(out)
    result = parse_out(out)
    return result


def run_and_record_one(
    result_table,
    **run_kwargs,
):
    cmd = cmd_template.format(
        **run_kwargs
    )
    print(cmd)
    result = run_cmd(cmd)

    if result is not None:
        result_table.add_row(
            **run_kwargs,
            **result
        )
        result_table.dump()


def run():
    result_table = CsvResultTable(
        cols=["tp", "use_dense_moe", "input_len", "output_len", "bs", "latency", "model_latency"],
        output_file="/mnt/user_storage/output.csv"
    )

    for tp, use_dense_moe, input_len, output_len, bs in itertools.product(
            TPS, USE_DENSE_MOES, INPUT_LEN, OUTPUT_LEN, BATCH_SIZE):
        run_and_record_one(
            result_table=result_table,
            tp=tp,
            use_dense_moe=use_dense_moe,
            input_len=input_len,
            output_len=output_len,
            bs=bs
        )


if __name__ == "__main__":
    run()
