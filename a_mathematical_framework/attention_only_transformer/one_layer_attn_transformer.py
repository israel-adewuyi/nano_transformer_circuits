import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch import Tensor
from jaxtyping import Float
from attention_only_transformer.attn_transformer import Embedding, Unembedding, AttentionLayer, Model

class OneLayerTransformer(Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = Embedding(cfg)
        self.attention_layer = AttentionLayer(cfg)
        self.LN = nn.LayerNorm(cfg.n_embd)
        self.umembedding = Unembedding(cfg)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                tokens: Float[Tensor, "batch seq_len d_model"],
                targets: Float[Tensor, "batch seq_len"] = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        tok_embed = self.embedding(tokens)

        residual = self.LN(self.attention_layer(tok_embed))

        logits = self.umembedding(residual)

        loss = None
        
        if targets is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss