import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Int



class Embedding(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(d_vocab, d_model)

    def forward(self, x: Int[Tensor, "batch seq"]):
        return self.embedding(x)



if __name__ == '__main__':
    embed = Embedding(500, 32)

    toks = torch.randint(0, 500, (5, 23))

    emb = embed(toks)
    print(emb)
    print(f'Shape of embed is {emb.shape}')
    # print(embed(toks))