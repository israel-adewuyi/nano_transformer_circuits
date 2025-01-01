import torch

from torch import Tensor
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

# https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/utils/hook_utils.py#L60

def addition_hook_fn(
    acts: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    refusal_dir: Float[Tensor, "d_model"]
):
    acts += (refusal_dir * 1.0)
    return acts

def ablation_hook_fn(
    acts: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    refusal_dir: Float[Tensor, "d_model"],
):
    refusal_dir = refusal_dir / (refusal_dir.norm(dim=-1, keepdim=True) + 1e-8)
    acts -= (acts @ refusal_dir).unsqueeze(-1) * refusal_dir
    return acts