import torch
import torch.nn as nn
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True

def conditional_linear(x, expert_indices, w, num_experts):
    if expert_indices.shape[0] <= 2:
        w_weights = w[expert_indices].view(-1, *w.shape[-2:]) # [T, A, O, I]
        return torch.einsum("ti, toi -> to", x, w_weights)
    else:
        dense_out = torch.einsum("ti, eoi -> teo", x, w)
        one_hot_indices = torch.nn.functional.one_hot(expert_indices.view(-1), num_classes=num_experts).to(dtype=dense_out.dtype)
        return torch.einsum("teo, te -> to", dense_out, one_hot_indices)

def conditional_feed_forward(x, expert_indices, w1, w2, w3, num_experts):
    x = x.unsqueeze(1).expand(x.shape[0], expert_indices.shape[-1], x.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    x1 = torch.nn.functional.silu(conditional_linear(x, expert_indices, w1, num_experts))
    x3 = conditional_linear(x, expert_indices, w3, num_experts)
    return conditional_linear((x1 * x3), expert_indices, w2, num_experts)

def dense_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    num_experts = w1.shape[0]

    expert_weights = torch.nn.functional.softmax(gating_output, dim=-1)
    expert_weights, expert_indices = torch.topk(expert_weights, topk, dim=-1) # [T, A], [T, A]
    expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # [T, A]

    expert_outs = conditional_feed_forward(hidden_states, expert_indices, w1, w2, w3, num_experts)
    return torch.einsum('tai,ta -> ti', expert_outs.view(-1, topk, expert_outs.shape[-1]), expert_weights)
