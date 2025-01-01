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


def get_all_layers_resid_acts(
    model: HookedTransformer, tokens: Float[Tensor, "batch seq_len"], post_instruct_pos: Int
) -> Float[Tensor, "n_layers pos d_model"]:
    """
    see - https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html#transformer_lens.ActivationCache.ActivationCache.stack_activation

    Run the model, caches the activations at each layer and each component and stack the activation for each layer, for each post instruction token position of the post residual stream. 

    Args:
        model (HookedTransformer): The transformer model used for forward pass.
        tokens (Float[Tensor]): A tensor of tokenized inputs
        post_instruct_pos: The index from the last pos, where the post instruction token starts

    Returns:
        Tensor: A tensor of shape (layers, post_instructin_token_positions, d_model) containing the tokenized sequences.
    
    """
    _, cache = model.run_with_cache(
        tokens
    )

    acts = cache.stack_activation(
        activation_name = "resid_post"
    )
    # Shape of acts is [layers, batch, seq_len, d_model]
    del cache 

    acts = acts[:, :, -post_instruct_pos:, :]
    # Take the mean across batches (prompts)
    acts = acts.mean(dim=-3)
    
    return acts