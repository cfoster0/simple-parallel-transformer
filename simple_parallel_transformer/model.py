import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat
from opt_einsum import contract
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    """Class for keeping track of config variables."""
    depth: int
    heads: int
    d_model: int
    vocab_size: int
    expansion_factor: int = 2
    max_seq_len: int = 2048
    seed: int = 10101
        
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)



class Block(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for a transformer-like block. It builds on
        several new ideas since the original "Attention is all you need":
        (1) Cogview's Sandwich Layer Norm, which puts a LayerNorm at the start 
            and end of the block
        (2) Single-block design from Gated Attention Unit, by way of Mamba.
            Instead of a separate attention and feedforward layer, combines
            them in parallel as a gated attention layer, with the gating being
            passed into a SiLu/SwiGLU activation function. Also expands the
            internal dimension to be larger than the residual dimension.
        (3) Smeared keys on each head, to facilitate learning of previous-token
            heads, induction heads, and n-grams.
        (4) per-head linear biases on the attention logits from ALiBi; half of heads are initialized as nonspatial

        ---
        References:
        Sandwich Layer Norm - https://arxiv.org/abs/2105.13290
        GAU - https://arxiv.org/abs/2202.10447
        Mamba - https://arxiv.org/abs/2312.00752
        Smeared Key - https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
        ALiBi - https://arxiv.org/abs/2108.12409
        """
        super(Block, self).__init__()
        self.heads = config.heads
        self.d_model = config.d_model
        self.d_expanded = self.d_model * config.expansion_factor
        self.d_head = (self.d_expanded) // self.heads
        
        l = config.max_seq_len
        
        self.in_ln = nn.LayerNorm(self.d_model)
        self.out_ln = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_expanded * 4, bias=False)
        self.out_proj = nn.Linear(self.d_expanded, self.d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02/sqrt(2 * config.depth))
        slopes = torch.cat([torch.logspace(0., -8., steps=self.heads//2, base=2.), torch.zeros(self.heads - (self.heads//2))], dim=0)
        mask = (slopes[:, None, None] * torch.arange(l)[None, None, :]) + torch.triu(-1e10 * torch.ones((l, l)), diagonal=1)[None, :, :]
        self.register_buffer('mask', mask)
        self.smear_factor = nn.Parameter(torch.linspace(-6., 6., self.heads))
        self.log_scale = nn.Parameter(torch.zeros(self.heads))
        
    def forward(self, x):
        b, l, d = x.shape

        q, k, v, p = torch.split(self.in_proj(self.in_ln(x)), [
                                   self.d_expanded,
                                   self.d_expanded,
                                   self.d_expanded,
                                   self.d_expanded,
                                   ], -1)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b h i d", h=self.heads), (q, k, v))
        smear = F.sigmoid(self.smear_factor)[None, :, None, None]
        k = ((1 - smear) * k) + (smear * F.pad(k, (0, 0, 1, -1)))
        s = torch.exp(self.log_scale)[None, :, None, None]
        mask = repeat(self.mask[:, :l, :l], "h i j -> b h i j", b=b)
        o = F.scaled_dot_product_attention(q / s, k / s, v, attn_mask=mask)
        o = rearrange(o, "b h i d -> b i (h d)")
        return self.out_ln(self.out_proj(F.silu(p) * o))

class Transformer(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for the model as a whole.
        """
        super(Transformer, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.sos_index = config.vocab_size
        
        self.embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for i in range(config.depth)])
        self.unembed = nn.Sequential(*[
                                  nn.LayerNorm((config.d_model)),
                                  nn.Linear(config.d_model, config.vocab_size, bias=True),
                                  ])

    def forward(self, x):
        x = self.embed(x)
        for (i, block) in enumerate(self.blocks):
            x = x + block(x)
        return self.unembed(x)
