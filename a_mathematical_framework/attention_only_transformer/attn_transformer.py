import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Int, Float
from dataclasses import dataclass
from config import Config

class Embedding(nn.Module):
    def __init__(self, cfg: Config):
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(self.cfg.d_vocab, self.cfg.d_model)

    def forward(self, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq_len d_model"]:
        return self.embedding(tokens)


class PositionalEmbedding(nn.Module):
    """
        RoPE
        Source - https://karthick.ai/blog/2024/Rotatory-Position-Embedding-(RoPE)/
    """
    def __init__(self, cfg: Config):
        super(PositionalEmbedding, self).__init__()
        self.cfg = cfg
        self.angle_rates = 1 / torch.pow(10000, torch.arange(0, self.cfg.d_model, 2).float() / self.cfg.d_model)
        self.angles = (torch.arange(self.cfg.context_size).unsqueeze(1) * self.angle_rates.unsqueeze(0))
        self.position_encodings = torch.stack((self.angles.cos(), self.angles.sin()), dim=2).flatten(1)

    def forward(self, embeddings: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        cos_enc, sin_enc = self.position_encodings[..., 0::2], self.position_encodings[..., 1::2]
        
        embeddings[..., 0::2] = embeddings[..., 0::2] * cos_enc - embeddings[..., 1::2] * sin_enc
        embeddings[..., 1::2] = embeddings[..., 1::2] * cos_enc + embeddings[..., 0::2] * sin_enc
        return embeddings
        



if __name__ == '__main__':
    cfg = Config()

    embed = Embedding(cfg)
    posembed = PositionalEmbedding(cfg)

    toks = torch.randint(0, 500, (5, 24))

    emb = embed(toks)
    print(emb)
    print(f'Shape of embed is {emb.shape}')
    print(posembed(emb))




    # print(embed(toks))