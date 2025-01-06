import torch
import einops
import transformer_lens as TLens
import plotly.graph_objects as go

from torch import Tensor
from typing import List
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

from utils.model_utils import (
    get_tokens,
    get_all_layers_resid_acts,
    compute_induce_score,
    compute_bypass_score,
    compute_KL_score,
    plot_refusal_metric,
    generate_responses
)

model_family = "google"

MODEL_DATA = {
    "google" : {
        "chat_prefix" : "<start_of_turn>user\n{}",
        "chat_suffix" : "<end_of_turn>\n<start_of_turn>model\n",
        "refusal_toks": [235285]
    },
    "qwen" : {
        "chat_prefix" : "<|im_start|>user\n{}",
        "chat_suffix" : "<|im_end|>\n<|im_start|>assistant\n",
        "refusal_toks": [40, 2121]
    }
}

def load_dataset(dataset_type: str):
    with open(f"datasets/train_{dataset_type}_prompts.txt") as file:
        train_prompts = file.readlines()

    with open(f"datasets/val_{dataset_type}_prompts.txt") as file:
        val_prompts = file.readlines()

    return (train_prompts, val_prompts)


def select_direction(
    induce_score: Float[Tensor, "pos n_layers"],
    bypass_score: Float[Tensor, "pos n_layers"]
):
    threshold = -7.5

    mask = induce_score > threshold

    # print(mask)

    filtered_induce_score = torch.where(mask, bypass_score, torch.tensor(float('inf')))

    idx = torch.argmin(filtered_induce_score)

    # print(idx)

    X_idx = idx // induce_score.shape[1]
    Y_idx = idx % induce_score.shape[1]

    return X_idx, Y_idx

if __name__ == "__main__":
    # Load the datasets from dir
    train_harmful_prompts, val_harmful_prompts = load_dataset(dataset_type="harmful")
    train_harmless_prompts, val_harmless_prompts = load_dataset(dataset_type="harmless")

    print(len(train_harmful_prompts), len(train_harmless_prompts))

    for model_name in ["gemma-2b-it", "gemma-2-2b-it"]:
        model = HookedTransformer.from_pretrained(model_name, device="cuda:1")

        # Get the tokens for the train dataset
        harmful_train_tokens = get_tokens(model, 
                                          train_harmful_prompts[:64], 
                                          MODEL_DATA[model_family]["chat_prefix"],
                                          MODEL_DATA[model_family]["chat_suffix"]
                                         )
        harmless_train_tokens = get_tokens(model, 
                                           train_harmless_prompts[:64],
                                           MODEL_DATA[model_family]["chat_prefix"],
                                           MODEL_DATA[model_family]["chat_suffix"]
                                          )

        post_instruct_pos = model.to_tokens(MODEL_DATA[model_family]["chat_suffix"]).shape[1] - 1

        print(f"Post instruction positions are {post_instruct_pos}")

        harmful_acts = get_all_layers_resid_acts(model, harmful_train_tokens, post_instruct_pos)
        harmless_acts = get_all_layers_resid_acts(model, harmless_train_tokens, post_instruct_pos)

        assert (harmful_acts.shape == harmless_acts.shape), "Both activations on train set should have the same shape"
        assert (harmful_acts.shape == (model.cfg.n_layers, post_instruct_pos, model.cfg.d_model))

        candidate_vectors = harmful_acts - harmless_acts

        candidate_vectors = einops.rearrange(
            candidate_vectors,
            "layers pos d_model -> pos layers d_model"
        )

        torch.save(candidate_vectors, "candidate_vectors.pt")

        assert(candidate_vectors.shape == (post_instruct_pos, model.cfg.n_layers, model.cfg.d_model))

        print("Computing induce score...")
        induce_score = compute_induce_score(model, candidate_vectors, val_harmless_prompts, MODEL_DATA[model_family]["refusal_toks"]).to('cpu')
        assert (induce_score.shape == (post_instruct_pos, model.cfg.n_layers)), "Induce score tensor shapes not correct"

        print("Computing bypass score...")
        bypass_score = compute_bypass_score(model, candidate_vectors, val_harmful_prompts, MODEL_DATA[model_family]["refusal_toks"]).to('cpu')
        assert (bypass_score.shape == (post_instruct_pos, model.cfg.n_layers)), "Bypass score tensor shapes not correct"

        X_idx, Y_idx = select_direction(induce_score, bypass_score)
        
        print(X_idx, Y_idx)

        generate_responses(model, candidate_vectors[X_idx][Y_idx])

        # print("Computing KL score")
        # KL_score = compute_KL_score(model, candidate_vectors, val_harmless_prompts)
        # assert (KL_score.shape == (post_instruct_pos, model.cfg.n_layers))

        # plot_refusal_metric(induce_score,"Induce score", "induce_score", model_name)
        # plot_refusal_metric(bypass_score,"Bypass score", "bypass_score", model_name)
        # plot_refusal_metric(KL_score,"KL score", "kl_score", model_name)

        