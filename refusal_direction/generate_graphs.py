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
    compute_bypass_score
)

MODEL_DATA = {
    "google" : {
        "chat_prefix" : "<start_of_turn>user\n{}",
        "chat_suffix" : "<end_of_turn>\n<start_of_turn>model\n",
        "refusal_toks": [235285]
    },
    "Qwen" : {
        "chat_prefix" : "<start_of_turn>user\n{}",
        "chat_suffix" : "<end_of_turn>\n<start_of_turn>model\n",
        "refusal_toks": [40, 2121]
    }
}

def load_dataset(dataset_type: str):
    with open(f"train_{dataset_type}_prompts.txt") as file:
        train_prompts = file.readlines()

    with open(f"val_{dataset_type}_prompts.txt") as file:
        val_prompts = file.readlines()

    return (train_prompts, val_prompts)

if __name__ == "__main__":
    # Load the datasets from dir
    train_harmful_prompts, val_harmful_prompts = load_dataset(dataset_type="harmful")
    train_harmless_prompts, val_harmless_prompts = load_dataset(dataset_type="harmless")

    print(len(train_harmful_prompts), len(train_harmless_prompts))

    for model_name in ["google/gemma-2b-it"]:
        model_family = model_name.split('/')[0]
        model = HookedTransformer.from_pretrained(model_name, device="cuda:2")

        # Get the tokens for the train dataset
        harmful_train_tokens = get_tokens(model, 
                                          train_harmful_prompts, 
                                          MODEL_DATA[model_family]["chat_prefix"],
                                          MODEL_DATA[model_family]["chat_suffix"]
                                         )
        harmless_train_tokens = get_tokens(model, 
                                           train_harmless_prompts,
                                           MODEL_DATA[model_family]["chat_prefix"],
                                           MODEL_DATA[model_family]["chat_suffix"]
                                          )

        # print(harmful_train_tokens.shape, harmless_train_tokens.shape)

        post_instruct_pos = model.to_tokens(MODEL_DATA[model_family]["chat_suffix"]).shape[1] - 1

        harmful_acts = get_all_layers_resid_acts(model, harmful_train_tokens, post_instruct_pos)
        harmless_acts = get_all_layers_resid_acts(model, harmless_train_tokens, post_instruct_pos)

        assert (harmful_acts.shape == harmless_acts.shape), "Both activations on train set should have the same shape"
        assert (harmful_acts.shape == (model.cfg.n_layers, post_instruct_pos, model.cfg.d_model))

        candidate_vectors = harmful_acts - harmless_acts

        candidate_vectors = einops.rearrange(
            candidate_vectors,
            "layers pos d_model -> pos layers d_model"
        )

        assert(candidate_vectors.shape == (post_instruct_pos, model.cfg.n_layers, model.cfg.d_model))

        print("Computing induce score...")
        induce_score = compute_induce_score(model, candidate_vectors, val_harmless_prompts, MODEL_DATA[model_family]["refusal_toks"])
        assert (induce_score.shape == (post_instruct_pos, model.cfg.n_layers)), "Induce score tensor shapes not correct"

        print("Computing bypass score")
        bypass_score = compute_bypass_score(model, candidate_vectors, val_harmful_prompts, MODEL_DATA[model_family]["refusal_toks"])
        assert (bypass_score.shape == (post_instruct_pos, model.cfg.n_layers)), "Bypass score tensor shapes not correct"

        print(induce_score)
    
    
    