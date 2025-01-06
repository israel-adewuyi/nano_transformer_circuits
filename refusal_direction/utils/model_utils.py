import torch
import einops
import functools
import transformer_lens as TLens
import torch.nn.functional as F
import plotly.graph_objects as go

from typing import List
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from utils.hook_utils import ablation_hook_fn, addition_hook_fn


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
        activation_name = "resid_pre"
    )
    # Shape of acts is [layers, batch, seq_len, d_model]
    del cache 

    acts = acts[:, :, -post_instruct_pos:, :]

    # Take the mean across batches (prompts)
    acts = acts.mean(dim=-3)
    
    return acts


def compute_induce_score(
    model: HookedTransformer,
    candidate_vectors: Float[Tensor, "pos layers d_model"],
    prompts: List[str],
    refusal_toks: List[Int]
) -> Float[Tensor, "pos n_layers"]:
    scores = torch.empty(candidate_vectors.shape[0], candidate_vectors.shape[1])
    
    for i in range(candidate_vectors.shape[0]):
        for j in range(candidate_vectors.shape[1]):
            temp_fn = functools.partial(addition_hook_fn, refusal_dir=candidate_vectors[i][j])
            logits = model.run_with_hooks(
                prompts,
                return_type="logits",
                fwd_hooks = [
                    (TLens.utils.get_act_name("resid_pre", j), temp_fn)
                ]
            )
            logits = logits[:, -1, :]

            induce_score = compute_refusal_metric(logits, refusal_toks)

            scores[i][j] = induce_score.item()
    return scores

def compute_bypass_score(
    model: HookedTransformer, 
    candidate_vectors: Float[Tensor, "pos layers d_model"],
    prompts: List[str],
    refusal_toks: List[Int]
) -> Float[Tensor, "pos n_layers"]:
    scores = torch.zeros(candidate_vectors.shape[0], candidate_vectors.shape[1])

    for i in range(candidate_vectors.shape[0]):
        for j in range(candidate_vectors.shape[1]):
            temp_fn = functools.partial(ablation_hook_fn, refusal_dir=candidate_vectors[i][j])
            logits = model.run_with_hooks(
                prompts,
                return_type="logits",
                fwd_hooks = [
                    (TLens.utils.get_act_name("resid_pre", k), temp_fn) for k in range(candidate_vectors.shape[1])
                ]  
                    # for k in range(candidate_vectors.shape[1])
            )
            logits = logits[:, -1, :]

            induce_score = compute_refusal_metric(logits, refusal_toks)

            scores[i][j] = induce_score.item()
    return scores

def compute_KL_score(
    model: HookedTransformer, 
    candidate_vectors: Float[Tensor, "pos layers d_model"],
    prompts: List[str],
):
    scores = torch.zeros(candidate_vectors.shape[0], candidate_vectors.shape[1]).to('cpu')
    
    baseline_logits = model(
        prompts,
        return_type="logits"
    )

    baseline_logits = baseline_logits[:, -1, :]

    baseline_logits = baseline_logits.to('cpu')

    for i in range(candidate_vectors.shape[0]):
        for j in range(candidate_vectors.shape[1]):
            print(f"Computing the KL at pos {i} and layer {j}")
            temp_fn = functools.partial(ablation_hook_fn, refusal_dir=candidate_vectors[i][j])
            logits = model.run_with_hooks(
                prompts,
                return_type="logits",
                fwd_hooks = [
                    (TLens.utils.get_act_name("resid_pre", j), temp_fn)
                ]
            )

            logits = logits[:, -1, :]

            logits = logits.to('cpu')

            kl_score = compute_KL_div_score(baseline_logits, logits)
            scores[i][j] = kl_score

    return scores


def compute_KL_div_score(
    baseline_logits: Float[Tensor, "batch seq_len d_model"],
    ablated_logits: Float[Tensor, "batch seq_len d_model"]
):
    probs1 = F.softmax(baseline_logits, dim=-1)
    probs2 = F.softmax(ablated_logits, dim=-1)
    
    log_probs1 = torch.log(probs1 + 1e-10)
    
    kl_div = F.kl_div(log_probs1, probs2, reduction='none', log_target=False)
    kl_div = kl_div.sum(dim=-1)  
    
    kl_div_mean = kl_div.mean()

    return kl_div_mean

def compute_refusal_metric(
    logits: Float[Tensor, "batch d_vocab"],
    refusal_toks: List[Int]
):
    epsilon = 1e-8
    probability_distribution = F.softmax(logits, dim=-1)
    # print(f"Shape of vocab is {probability_distribution.shape}")

    refusal_probs = probability_distribution[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    score = torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

    return score.mean()


def plot_refusal_metric(
    scores: Float[Tensor, "pos n_layers"],
    title: str,
    plot_title: str,
    model_family: str
):
    positions, layers = scores.shape
    scores = scores.numpy()
    
    fig = go.Figure()

    for i, name in zip(range(positions), ['<end_of_turn>', "newline", '<start_of_turn>', 'model', "newline"]):
        fig.add_trace(go.Scatter(
            x=list(range(layers)),
            y=scores[i],
            mode='lines+markers',
            name=name,
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Source layer',
        yaxis_title="Refusal score",
        legend_title="source position"
    )
    
    # Save the figure as an image file
    fig.write_image(f"artefacts/{model_family}/{plot_title}.png")