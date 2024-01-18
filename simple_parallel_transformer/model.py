import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from einops import rearrange
from opt_einsum import contract
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    """Class for keeping track of config variables."""
    depth: int
    heads: int
    d_head: int
    vocab_size: int
    max_seq_len: int = 2048
    expansion_factor: int = 4
    seed: int = 10101
        
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)


class LearnedALiBi(nn.Module):
    def __init__(self, heads):
        super().__init__()
        slopes = torch.cat([torch.logspace(0., -6., steps=heads//2, base=2.), torch.zeros(heads - (heads//2))], dim=0)
        self.slopes = nn.Parameter(rearrange(slopes, 'h -> () h () ()'))

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        bias = rearrange(torch.arange(j, device = device), 'j -> () () () j') * self.slopes
        bias = F.pad(bias, (0, 0, 0, 0, 0, h - bias.shape[1]))
        return qk_dots + bias
    
class Attention(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for a transformer-like attention block. It 
        builds on some new ideas since the original "Attention is all you need":
        (1) Smeared keys on each head, to facilitate learning of previous-token
        heads, induction heads, and n-grams.
        (2) learned per-head linear biases on the attention logits similar 
            to ALiBi; half of heads are initialized as nonspatial

        ---
        References:
        Smeared Key - https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
        ALiBi - https://arxiv.org/abs/2108.12409
        """
        super(Attention, self).__init__()
        self.heads = config.heads
        self.d_head = config.d_head
        self.d_model = self.heads * self.d_head
        
        self.ln_1 = nn.LayerNorm(self.d_model)
        self.ln_2 = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_model * 3, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02/sqrt(2 * config.depth))
        causal_mask = torch.tril(torch.ones((config.max_seq_len, config.max_seq_len)))
        self.register_buffer('causal_bias', rearrange(-1e10 * (1. - causal_mask), "i j -> () () i j"))
        self.alibi = LearnedALiBi(self.heads)
        self.smear_factor = nn.Parameter(torch.linspace(-6., 6., self.heads))
        
    def forward(self, x):
        b, l, d = x.shape

        q, k, v = torch.split(self.in_proj(self.ln_1(x)), [
                                   self.d_model,
                                   self.d_model,
                                   self.d_model,
                                   ], -1)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b h i d", h=self.heads), (q, k, v))
        smear = F.sigmoid(self.smear_factor)[None, :, None, None]
        k = ((1 - smear) * k) + (smear * F.pad(k, (0, 0, 1, -1)))
        logits = contract("b h i d, b h j d -> b h i j", q, k) * (self.d_head ** -0.5)
        a = F.softmax(self.causal_bias[..., :l, :l] + self.alibi(logits), dim=-1)
        o = rearrange(contract("b h i j, b h j d -> b h i d", a, v), "b h i d -> b i (h d)")
        return self.ln_2(self.out_proj(o))

class MLP(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for a transformer-like MLP block. It uses a 
        SiLU gated linear unit (also known as SwiGLU) to replace the 
        traditional ReLu/GELU nonlinearity in the MLP

        ---
        References:
        GLU - https://arxiv.org/abs/2002.05202v1
        """
        super(MLP, self).__init__()
        self.d_model = config.heads * config.d_head
        self.d_bilinear = (self.d_model * config.expansion_factor) // 2
        
        self.ln_1 = nn.LayerNorm(self.d_model)
        self.ln_2 = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_bilinear * 2, bias=False)
        self.out_proj = nn.Linear(self.d_bilinear, self.d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02/sqrt(2 * config.depth))
        
    def forward(self, x):
        p1, p2 = torch.split(self.in_proj(self.ln_1(x)), [
                                   self.d_bilinear,
                                   self.d_bilinear,
                                   ], -1)
        return self.ln_2(self.out_proj(torch.cat([F.silu(p1) * p2], dim=-1)))

class Transformer(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for the model as a whole.
        """
        super(Transformer, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.sos_index = config.vocab_size
        d_model = config.heads * config.d_head
        
        self.embed = nn.Embedding(config.vocab_size + 1, d_model)
        self.attentions = nn.ModuleList([Attention(config) for i in range(config.depth)])
        self.mlps = nn.ModuleList([MLP(config) for i in range(config.depth)])
        self.unembed = nn.Sequential(*[
                                  nn.LayerNorm((d_model)),
                                  nn.Linear(d_model, config.vocab_size, bias=True),
                                  ])

    def forward(self, x):
        x = self.embed(x)
        for (i, (attention, mlp)) in enumerate(zip(self.attentions, self.mlps)):
            x = x + attention(x)
            x = x + mlp(x)
        return self.unembed(x)
