import torch
import einops
import torch.nn as nn

from torch import Tensor
from yacs.config import CfgNode as CN
from jaxtyping import Int, Float
from dataclasses import dataclass

class Embedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)

    def forward(self, tokens: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        return self.embedding(tokens)

class Unembedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.W_U = nn.Parameter(torch.empty(self.cfg.n_embd, self.cfg.vocab_size))
        nn.init.xavier_normal_(self.W_U)
        self.b_U = nn.Parameter(torch.zeros((self.cfg.vocab_size), requires_grad=False))

    def forward(self, normalized_resid_final: Float[Tensor, "batch seq_len d_model"])-> Float[Tensor, "batch seq_len d_vocab"]:
        return einops.einsum(
            self.W_U, normalized_resid_final,
            "d_model d_vocab, batch seq_len d_model -> batch seq_len d_vocab"
        ) + self.b_U
        

class PositionalEmbedding(nn.Module):
    """
        RoPE
        Source - https://karthick.ai/blog/2024/Rotatory-Position-Embedding-(RoPE)/
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.angle_rates = 1 / torch.pow(10000, torch.arange(0, self.cfg.n_embd, 2).float() / self.cfg.n_embd)
        self.angles = (torch.arange(self.cfg.block_size).unsqueeze(1) * self.angle_rates.unsqueeze(0))
        self.position_encodings = torch.stack((self.angles.cos(), self.angles.sin()), dim=2).flatten(1).to(cfg.device)

    def forward(self, embeddings: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        cos_enc, sin_enc = self.position_encodings[..., 0::2], self.position_encodings[..., 1::2]
        
        embeddings[..., 0::2] = embeddings[..., 0::2] * cos_enc - embeddings[..., 1::2] * sin_enc
        embeddings[..., 1::2] = embeddings[..., 1::2] * cos_enc + embeddings[..., 0::2] * sin_enc
        return embeddings
        
class AttentionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=self.cfg.n_embd,
            num_heads=self.cfg.n_head,
            batch_first=True,
            bias=False
        )

        self.pos_embed = PositionalEmbedding(self.cfg)

        self.LN = nn.LayerNorm(self.cfg.n_embd)

    def forward(self, tok_embed: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"] :
        normed_embed = self.LN(tok_embed)

        normed_pos_embed = self.pos_embed(normed_embed)

        mask = torch.triu(torch.ones(tok_embed.shape[1], tok_embed.shape[1]), diagonal=1).bool().to(self.cfg.device)

        attn_out, _ = self.attn_layer(query=normed_pos_embed, key=normed_pos_embed, value=normed_embed, attn_mask=mask)

        return tok_embed + attn_out


class Model(nn.Module):
    @staticmethod
    def get_default_config():
        C = CN()

        C.vocab_size = None
        C.block_size = None

        # model dimensions
        C.n_embd = 768
        C.n_head = 12

        # dropout hyperparameters
        C.pos_embd_pdrop = 0.0
        C.attn_pdrop = 0.0

        return C
        
    def __init__(self):
        super().__init__()

    def forward(self, 
                tokens: Float[Tensor, "batch seq_len d_model"],
                targets: Float[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """Forward pass"""
        raise NotImplementedError

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        unembedding_layer_name = 'unembedding'
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.startswith('umembedding'):
                    # explicitly add unembedding layer weights to decay
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx