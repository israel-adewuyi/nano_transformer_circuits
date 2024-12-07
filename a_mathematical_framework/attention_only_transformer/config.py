from dataclasses import dataclass

@dataclass
class Config:
    n_head: int = 12
    d_head: int = 64
    d_model: int = 768
    context_size: int = 24
    d_vocab: int = 50276