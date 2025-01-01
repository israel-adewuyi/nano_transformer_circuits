import torch

from typing import List
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer


def get_tokens(
    model: HookedTransformer, prompts: List[str], prefix: str, suffix: str
) -> Float[Tensor, "batch seq_len d_model"]:
    """
    Converts a list of prompts into tokenized sequences with specified model's chat template prefix and suffix.

    The function separately tokenizes (prefix-prompt) and suffix to make it easy to get the post instruction tokens.

    Args:
        model (HookedTransformer): The transformer model used for tokenization.
        prompts (List[str]): A list of input prompts to be tokenized.
        prefix (str): A prefix string to be formatted with each prompt.
        suffix (str): A suffix string to be appended to each prompt.

    Returns:
        Tensor: A tensor of shape (batch, seq_len, d_model) containing the tokenized sequences.
    """
    appended_prompts = [prefix.format(prompt) for prompt in prompts]
    suffixes = [suffix for _ in prompts]

    tokens =  model.to_tokens(appended_prompts)
    suf_tokens = model.to_tokens(suffixes)[:, 1:]

    tokens = torch.cat([tokens, suf_tokens], dim=-1)

    return tokens