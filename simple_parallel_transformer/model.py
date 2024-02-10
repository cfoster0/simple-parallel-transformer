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


def linear_attention(q, k, v, mask, eps=1e-5):
    """
    Apply causal linear attention to q, k, and v, with a multiplicative mask that can be used for ALiBi-like decay.
    """
    attn = torch.tril(contract("... i d , ... j d -> ... i j", q, k) * mask)
    norm = 1. / (attn.sum(dim=-1) + eps)
    return contract("... i j, ... i , ... j d -> ... i d", attn, norm, v)

class Block(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for a transformer-like block. It builds on
        several new ideas since the original "Attention is all you need":
        (1) Cogview's Sandwich Layer Norm, which puts a LayerNorm at the start 
            and end of the block
        (2) Single-block design from Gated Attention Unit, by way of Mamba.
            Instead of a separate feedforward layer, combines
            them in parallel as a gated mixer layer, with the gating being
            passed into a SiLu/SwiGLU activation function. Also expands the
            internal dimension to be larger than the residual dimension.
        (3) Smeared keys on each head, to facilitate learning of previous-token
            heads, induction heads, and n-grams.
        (4) attention replaced with linear attention with learned feature maps with "spiky" softmax activations,
            inspired by Hedgehog; implicitly folded into the query & key projections, each head is initialized with zeroed weights & biases
        (5) per-head exponential decay on the linear attention, similar to ALiBi; half of heads are initialized as nonspatial

        ---
        References:
        Sandwich Layer Norm - https://arxiv.org/abs/2105.13290
        GAU - https://arxiv.org/abs/2202.10447
        Mamba - https://arxiv.org/abs/2312.00752
        Smeared Key - https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
        Hedgehog - https://arxiv.org/abs/2402.04347
        ALiBi - https://arxiv.org/abs/2108.12409
        """
        super(Block, self).__init__()
        self.heads = config.heads
        self.d_model = config.d_model
        self.d_expanded = self.d_model * config.expansion_factor
        self.d_head = (self.d_expanded) // self.heads
        
        self.in_ln = nn.LayerNorm(self.d_model)
        self.q_ln = nn.LayerNorm(self.d_model)
        self.k_ln = nn.LayerNorm(self.d_model)
        self.out_ln = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_expanded * 2, bias=False)
        self.out_proj = nn.Linear(self.d_expanded, self.d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02/sqrt(2 * config.depth))
        self.smear_factor = nn.Parameter(torch.linspace(-6., 6., self.heads))
        self.log_scale = nn.Parameter((torch.ones(self.heads) * (self.d_head ** 0.25)).log())

        alibi = (torch.logspace(0, -8, self.heads // 2, base=2)[:, None, None] * -F.relu(torch.arange(config.max_seq_len)[:, None] - torch.arange(config.max_seq_len)[None, :])[None, :, :])
        self.register_buffer('mask', torch.cat([alibi, torch.zeros_like(alibi)]).exp())

        self.f_qproj = nn.Linear(self.d_model, self.d_expanded, bias=True)
        nn.init.zeros_(self.f_qproj.weight)
        nn.init.zeros_(self.f_qproj.bias)
        self.f_kproj = nn.Linear(self.d_model, self.d_expanded, bias=True)
        nn.init.zeros_(self.f_kproj.weight)
        nn.init.zeros_(self.f_kproj.bias)
        
    def forward(self, x):
        b, l, d = x.shape

        v, p = torch.split(self.in_proj(self.in_ln(x)), [
                                   self.d_expanded,
                                   self.d_expanded,
                                   ], -1)
        smear = F.sigmoid(self.smear_factor)[None, :, None, None]
        s = torch.exp(self.log_scale)[None, :, None, None]

        q = self.f_qproj(self.q_ln(x))
        k = self.f_kproj(self.k_ln(x))
        
        
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b h i d", h=self.heads), (q, k, v))
        
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)
        
        k = ((1 - smear) * k) + (smear * F.pad(k, (0, 0, 1, -1)))
        
        o = linear_attention(q/s, k/s, v, self.mask[:, :l, :l])
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

