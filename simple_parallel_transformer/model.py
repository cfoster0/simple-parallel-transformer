import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange

from opt_einsum import contract

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from .extra_modules import *

@dataclass
class Config:
    """Class for keeping track of config variables."""
    depth: int
    heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int = 2048
    seed: int = 10101
        
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)

def exists(val):
    return val is not None

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> () h () ()')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        import math
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]

        bias = torch.arange(j, device = device)
        bias = rearrange(bias, 'j -> () () () j')
        bias = bias * self.slopes
        bias = F.pad(bias, (0, 0, 0, 0, 0, h - bias.shape[1]))
        self.register_buffer('bias', bias, persistent = False)
        return qk_dots + self.bias
      
class Shift(nn.Module):
    def __init__(self, dimensions, n):
        super(Shift, self).__init__()
        self.dimensions = dimensions
        self.n = n

    def forward(self, x):
        part = x[..., :self.dimensions]
        x[..., :self.dimensions] = F.pad(part, (0, 0, self.n, -self.n), value=0)
        return x
    
class Residual(nn.Module):
    def __init__(self, residual):
        """
        In the constructor we stash way the module that'll be called along
        the residual branch. This is just for convenience.
        """
        super(Residual, self).__init__()
        self.residual = residual

    def forward(self, x, attn):
        x_residual, attn_residual = self.residual(x, attn)
        return x + x_residual, attn + attn_residual

class Block(nn.Module):
    def __init__(self, config: Config, layer_depth: int):
        """
        In this module is the code for a transformer block. This builds on
        Ben Wang's ideas of (1) doing the attention and feedforward residual
        computations in parallel instead of in series, which incidentally lets
        them share a single layer norm, and (2) merging the Wq, Wk, & Wv
        projections with the first FF projection and the Wo projection with the
        second FF projection, which should simplify and speed up the design.
        For a reference on these methods, see the below link:
        https://github.com/kingoflolz/mesh-transformer-jax
        """
        super(Block, self).__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.hidden_dim = self.heads * self.head_dim
        self.max_seq_len = config.max_seq_len
        self.qkv_dim = self.hidden_dim * (3 + 2)
        self.vp_dim = self.hidden_dim * (1 + 1)

        self.in_ln = nn.LayerNorm(self.hidden_dim)
        self.mid_ln = nn.LayerNorm(self.hidden_dim * 2)
        self.q_ln = nn.LayerNorm(self.hidden_dim)
        self.k_ln = nn.LayerNorm(self.hidden_dim)

        self.out_ln = nn.LayerNorm(self.hidden_dim)
        self.shift = Shift(self.hidden_dim // 4, 1)
        self.in_proj = nn.Linear(self.hidden_dim, self.qkv_dim, False)
        nn.init.orthogonal_(self.in_proj.weight)
        self.out_proj = nn.Linear(self.vp_dim, self.hidden_dim, True)
        nn.init.zeros_(self.out_proj.weight)
        self.alibi = AlibiPositionalBias(self.heads)

        self.mixing_tensor = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        causal_mask = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len)))
        causal_bias = -1e10 * (1. - causal_mask)
        self.register_buffer('causal_bias', rearrange(causal_bias, "i j -> () () i j"))

    def forward(self, x, accumulated_logits):
        b, l, d = x.shape

        x = self.in_ln(x)
        x[..., :d//4] = self.shift(x[..., :d//4])
        y = self.in_proj(x)
        q, k, v, p = torch.split(y, [
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim * 2,
                                   ], -1)
        q = self.q_ln(q)
        k = self.k_ln(k)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b h i d", h=self.heads, b=b), (q, k, v))
        logits = einsum("b h i d, b h j d -> b h i j", q, k) * (self.head_dim ** -0.5)
        logit_residual = logits.mean(dim=1, keepdim=True)
        logits = self.causal_bias + self.alibi(accumulated_logits + logits)
        a = F.softmax(logits, dim=-1)
        o = einsum("b h i j, b h j d -> b h i d", a, v)
        o = rearrange(o, "b h i d -> b i (h d)")

        input = self.mid_ln(p)
        left = p[..., :self.hidden_dim]
        right = p[..., self.hidden_dim:]
        p = contract("...r, ...r, or -> ...o", left, right, self.mixing_tensor) * (self.hidden_dim ** -0.5)

        x = torch.cat([o, p], dim=-1)
        x = self.out_proj(x)
        x = self.out_ln(x)
        return x, logit_residual

class Transformer(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for the model as a whole, including the
        prelude (where tokens become hidden representations), the body (where
        hidden representations are transformed), and the postlude (where
        hidden representations become token distributions).
        """
        super(Transformer, self).__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        hidden_dim = config.heads * config.head_dim
        self.sos_index = config.vocab_size
        
        self.prelude = nn.Sequential(*[
                                  nn.Embedding(config.vocab_size + 1, hidden_dim),
                                  ])
        self.body = nn.Sequential(*[Residual(Block(config, layer_depth)) for layer_depth in range(config.depth)])
        self.postlude = nn.Sequential(*[
                                  nn.LayerNorm((hidden_dim)),
                                  nn.Linear(hidden_dim, config.vocab_size, bias=True),
                                  ])

    def forward(self, x):
        b, i = x.shape
        x = self.prelude(x)
        attn = torch.zeros((b, 1, i, i), device=x.device)
        for (i, layer) in enumerate(self.body):
            x, attn = layer(x, attn)
        x = self.postlude(x)
        return x
