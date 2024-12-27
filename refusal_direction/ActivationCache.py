import re
import torch

from copy import deepcopy
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class CacheManager:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.ActivationsCache = {}
        self.hooks = {}

        self.pattern = r'[a-z]+\.[a-z]+\.[0-9]+'
        self.layer_pattern = r'[0-9]+'

        self.device = device

    def get_activations(self, input):
        inputs = self.tokenizer(input, 
                 return_tensors="pt",
                 max_length=256,
                 padding=True,
                 truncation=True if 256 else False)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.model(**inputs)

        cache = deepcopy(self.ActivationsCache)
        self.clear_cache()
        return cache


    def _get_hook_function(self, layer_idx: int):
        def hook_function(module, input, output):
            # print(f"Function calls got here at layer {layer_idx}")
            if isinstance(output, tuple):
                self.ActivationsCache[layer_idx] = output[0].detach().cpu()
            else:
                self.ActivationsCache[layer_idx] = output.detach().cpu()
            # print(self.ActivationsCache[layer_idx].shape)

        return hook_function

    def attach_hooks(self, ):
        for (name, module) in self.model.named_modules():
            match = re.compile(self.pattern).fullmatch(name)

            if match:
                layer_idx = re.compile(self.layer_pattern).findall(name)
                hook_fn = self._get_hook_function(layer_idx[0])
                # print(f"Found module name --> {name}, at layer {layer_idx[0]}")
                self.hooks[layer_idx[0]] = module.register_forward_hook(hook_fn)

    def get_layer_names(self):
        return self.model.model.layers

    def cleanup(self):
        for hook in self.hooks.values():
            hook.remove()

        self.hooks.clear()

    def clear_cache(self):
        self.ActivationsCache.clear()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        # do something I am not sure of yet
        self.cleanup()