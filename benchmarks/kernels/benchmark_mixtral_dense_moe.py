import json
import os
import sys

import torch
import torch.nn.functional as F
import triton

from vllm.model_executor.layers.fused_moe.dense_moe import dense_moe

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.set_default_dtype(torch.bfloat16)

def main():
    method = torch.compile(dense_moe, fullgraph=True)
    bss = [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
            2048, 3072, 4096
    ]
    tps  = [4]
    for tp in tps:
        print("tp=", tp)
        for bs in bss:
            run_grid(bs, method=method, tp=tp)

    # from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
    # method = fused_moe
    # for bs in [
    #         1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
    # ]:
    #     run_grid(bs, method=method)
    


def run_grid(bs, method, tp=1):
    d_model = 4096
    num_total_experts = 8
    top_k = 2
    tp_size = tp
    model_intermediate_size = 14336
    num_layers = 32
    num_calls = 100

    num_warmup_trials = 1
    num_trials = 1

    # if bs <= 16:
    #     BLOCK_SIZES_M = [16]
    # elif bs <= 32:
    #     BLOCK_SIZES_M = [16, 32]
    # elif bs <= 64:
    #     BLOCK_SIZES_M = [16, 32, 64]
    # elif bs <= 128:
    #     BLOCK_SIZES_M = [16, 32, 64, 128]
    # else:
    #     BLOCK_SIZES_M = [16, 32, 64, 128, 256]
    #for block_size_n in [32, 64, 128, 256]:
    ## for block_size_n in [32]:
    #    for block_size_m in BLOCK_SIZES_M:
    #        # for block_size_k in [64]:
    #        for block_size_k in [64, 128, 256]:
    #            for group_size_m in [1, 16, 32, 64]:
    #            # for group_size_m in [1]:
    #                for num_warps in [4, 8]:
    #                # for num_warps in [4]:
    #                    configs.append({
    #                        "BLOCK_SIZE_M": block_size_m,
    #                        "BLOCK_SIZE_N": block_size_n,
    #                        "BLOCK_SIZE_K": block_size_k,
    #                        "GROUP_SIZE_M": group_size_m,
    #                        "num_warps": num_warps,
    #                        "num_stages": 4,
    #                    })

    configs = [{}]

    best_config = None
    best_time_us = 1e20

    for config in configs:
        # warmup
        # print('warming up')
        try:
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_calls,
                    bs=bs,
                    d_model=d_model,
                    num_total_experts=num_total_experts,
                    top_k=top_k,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    method=method,
                    config=config,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # trial
        # print('benchmarking')
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                num_total_experts=num_total_experts,
                top_k=top_k,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                method=method,
                config=config,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

            # print(f'{kernel_dur_us=:.1f} {model_dur_ms=:.1f}'
            #       f' {bs=} {tp_size=} {top_k=} {num_total_experts=} '
            #       f'{d_model=} {model_intermediate_size=} {num_layers=}')
    print(f"{bs}: {best_time_us:.0f}us")
    # print(f"best_time_us: {best_time_us}, tp: {tp_size}, bs: {bs}, config: {best_config}")

    # holds Dict[str, Dict[str, int]]
    filename = "temp.json"
    # print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def run_timing(num_calls: int, bs: int, d_model: int, num_total_experts: int,
               top_k: int, tp_size: int, model_intermediate_size: int, method,
               config) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    ws = torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w1s = torch.rand(
        (num_total_experts, shard_intermediate_size, d_model),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w3s = torch.rand(
        (num_total_experts, shard_intermediate_size, d_model),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2s = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gating_output = F.softmax(torch.rand(
        (num_calls, bs, num_total_experts),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ),
                              dim=-1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        method(
            hidden_states=hidden_states,
            w1=w1s,
            w2=w2s,
            w3=w3s,
            gating_output=gating_output[i],
            topk=2,
            renormalize=True,
            inplace=True,
            override_config=config,
        )
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    sys.exit(main())
