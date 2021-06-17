import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class Config:
    """Class for keeping track of config variables."""
    depth: int
    heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int = 2048
    expansion_factor: int = 4
        
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)


class Residual(nn.Module):
    def __init__(self, residual):
        """
        In the constructor we stash way the module that'll be called along
        the residual branch. This is just for convenience.
        """
        super(Residual, self).__init__()
        self.residual = residual

    def forward(self, x):
        return x + self.residual(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        """
        In the constructor we generate rotary positional embeddings. When called
        in forward(), we assume that the start and end positions are within the
        bounds of 0 and max_seq_len. For an intuitive look into rotary 
        positional embeddings, see the below link:
        https://blog.eleuther.ai/rotary-embeddings/
        """
        super().__init__()
        dim = config.head_dim
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config.max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('freqs', rearrange(emb, 'n d -> () n () d'))

    def rotate_half(self, x):
        x = rearrange(x, '... (j d) -> ... j d', j = 2)
        x1, x2 = x.unbind(dim = -2)
        return torch.cat((-x2, x1), dim = -1)

    def forward(self, x, start, end):
        freqs = self.freqs[:, start:end]
        return (x * freqs.cos()) + (self.rotate_half(x) * freqs.sin())

class Block(nn.Module):
    def __init__(self, config: Config):
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
        self.expansion_factor = config.expansion_factor
        self.qkvff_dim = self.hidden_dim * (3 + self.expansion_factor)
        self.vff_dim = self.hidden_dim * (1 + self.expansion_factor)

        self.ln = nn.LayerNorm(self.hidden_dim)
        self.in_proj = nn.Linear(self.hidden_dim, self.qkvff_dim, False)
        self.out_proj = nn.Linear(self.vff_dim, self.hidden_dim, False)
        self.rotary = RotaryEmbedding(config)

    def forward(self, x):
        b, l, d = x.shape

        x = self.ln(x)
        x = self.in_proj(x)
        q, k, v, ff = torch.split(x, [
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim * self.expansion_factor
                                   ], -1)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b i h d", h=self.heads), (q, k, v))
        (q, k) = map(lambda x: self.rotary(x, 0, l), (q, k))
        a = einsum("bihd,bjhd -> bhij", q, k) * (self.head_dim ** -0.5)
        causal_mask = torch.tril(torch.ones((l, l), device=a.device))
        causal_bias = -1e10 * (1. - causal_mask)
        a += rearrange(causal_bias, "i j -> () () i j")
        a = F.softmax(a, dim=-1)
        o = einsum("bhij,bjhd -> bihd", a, v)
        o = rearrange(o, "b i h d -> b i (h d)")
        ff = F.gelu(ff)
        x = torch.cat([o, ff], dim=-1)
        x = self.out_proj(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for the model as a whole, including the
        prelude (where tokens become hidden representations), the body (where
        hidden representations are transformed), and the postlude (where
        hidden representations become token distributions).
        """
        super(Transformer, self).__init__()
        self.max_seq_len = config.max_seq_len
        hidden_dim = config.heads * config.head_dim
        
        prelude = nn.Sequential(*[
                                  nn.Embedding(config.vocab_size, hidden_dim),
                                  ])
        body = nn.Sequential(*[Residual(Block(config)) for _ in range(config.depth)])
        postlude = nn.Sequential(*[
                                  nn.LayerNorm((hidden_dim)),
                                  nn.Linear(hidden_dim, config.vocab_size, True),
                                  ])

        network = nn.Sequential(*[
                                  prelude, 
                                  body, 
                                  postlude,
                                  ])

        self.network = network

    def forward(self, x):
        return self.network(x)
