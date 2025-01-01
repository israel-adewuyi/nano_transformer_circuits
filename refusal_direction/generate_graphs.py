import torch
import einops
import transformer_lens as TLens
import plotly.graph_objects as go

from torch import Tensor
from typing import List
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

from utils.model_utils import get_tokens

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

        print(harmful_train_tokens.shape, harmless_train_tokens.shape)

        


    

    

    





























    
    # with open("train_harmful_prompts.txt", "r") as harmful_file:
    #     train_harmful_prompts = harmful_file.readlines()
    #     # Add for val

    # with open("train_harmless_prompts.txt", "r") as harmless_file:
    #     train_harmless_prompts = harmless_file.readlines()
    #     # Add for val

    # """
    #     1. Getting logits
    #     --2. Concating all the tensors into [Layer, batch, seq_len, d_model]--
    #     3. Mean over batch dim
    #     --4. read seq_len out, so I can use in assert statement
    #     --5. What is the post token pos, how to get it for different models? 
    # """

    # train_harmful_prompts = append_chat_template(train_harmful_prompts, "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n")
    # train_harmless_prompts = append_chat_template(train_harmless_prompts, "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n")   

    # seq_len_harmful, seq_len_harmless = None, None
    # d_model = 2048
    # n_layers = 18

    # with CacheManager("google/gemma-2b-it", "cuda:3") as cachemanager:
    #     cachemanager.attach_hooks()
    
    #     harmful_activation_cache = cachemanager.get_activations(train_harmful_prompts)
    
    #     harmless_activation_cache = cachemanager.get_activations(train_harmless_prompts)

    #     seq_len_harmful = harmful_activation_cache['0'].shape[1]
    #     seq_len_harmless = harmless_activation_cache['0'].shape[1]
    
    #     print(harmful_activation_cache['1'].shape)
    #     print(harmless_activation_cache['1'].shape)


    # # Cache is of shape (nLayers, batch, seq_len, d_model)
    # harmful_activation_cache = torch.stack(list(harmful_activation_cache.values()), dim=0)
    # harmless_activation_cache = torch.stack(list(harmless_activation_cache.values()), dim=0)

    # assert harmful_activation_cache.shape == (n_layers, 128, seq_len_harmful, d_model)
    # assert harmless_activation_cache.shape == (n_layers, 128, seq_len_harmless, d_model)

    # mean_harmful_activation_cache = harmful_activation_cache.mean(dim=-3)
    # mean_harmless_activation_cache = harmless_activation_cache.mean(dim=-3)

    # post_mean_harmful_activation_cache = mean_harmful_activation_cache[:, -5:, :]
    # post_mean_harmless_activation_cache = mean_harmless_activation_cache[:, -5:, :]

    # print(f"Shape of post mean harmful activation cache is {post_mean_harmful_activation_cache.shape}")
    # print(f"Shape of post mean harmless activation cache is {post_mean_harmless_activation_cache.shape}")
    
    
    