import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from opt_einsum import contract
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
    seed: int = 10101
        
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)

def exists(val):
    return val is not None

def shift(x, dimensions, n):
  return torch.cat([F.pad(x[..., :dimensions], (0, 0, n, -n), value=0), x[..., dimensions:]], dim=-1)

class LearnedALiBi(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.slopes = nn.Parameter(rearrange(torch.logspace(0., -6., steps=heads, base=2.), 'h -> () h () ()'))
        self.register_buffer('bias', None, persistent = False)

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if (not self.training) and exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]

        bias = rearrange(torch.arange(j, device = device), 'j -> () () () j') * self.slopes
        bias = F.pad(bias, (0, 0, 0, 0, 0, h - bias.shape[1]))
        self.register_buffer('bias', bias, persistent = False)
        return qk_dots + bias
    
class Block(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for a transformer block. This builds on
        several ideas: (1) Ben Wang's parallel attention and feedforward block
        (2) Cogview's Sandwich Layer Norm, which puts a Layer Norm at the start 
        and end of the block (3) ALiBi-like per-head learned linear biases on 
        the attention similar to ALiBi (4) BlinkDL's token shift on a portion
        of dimensions to provide direct access to previous token representations
        & easy n-gram learning and (5) a home-grown modification to GeGLU that
        uses a single projection matrix and uses a rolled version of the original
        representation to do the linear modulation (6) a home-grown modification
        that adds a ReLU + LayerNorm after the initial projection & token shift.
        ---
        References:
        Parallel Block - https://github.com/kingoflolz/mesh-transformer-jax
        Sandwich Layer Norm - https://arxiv.org/abs/2105.13290
        ALiBi - https://arxiv.org/abs/2108.12409
        Token Shift - https://github.com/BlinkDL/RWKV-LM
        GeGLU - https://arxiv.org/abs/2002.05202v1
        """
        super(Block, self).__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.expansion_factor = config.expansion_factor
        self.hidden_dim = self.heads * self.head_dim

        qkvp_dim = self.hidden_dim * (3 + self.expansion_factor)
        vp_dim = self.hidden_dim * (1 + self.expansion_factor)

        self.in_ln = nn.LayerNorm(self.hidden_dim)
        self.mid_ln = nn.LayerNorm(qkvp_dim)
        self.out_ln = nn.LayerNorm(self.hidden_dim)

        self.in_proj = nn.Linear(self.hidden_dim, qkvp_dim, bias=False)
        self.out_proj = nn.Linear(vp_dim, self.hidden_dim, bias=False)
        nn.init.orthogonal_(self.in_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

        causal_mask = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len)))
        causal_bias = -1e10 * (1. - causal_mask)
        self.register_buffer('causal_bias', rearrange(causal_bias, "i j -> () () i j"))
        self.alibi = LearnedALiBi(self.heads)

    def forward(self, x):
        b, l, d = x.shape

        x = self.in_ln(x)
        x[..., :d//4] = shift(x[..., :d//4], self.hidden_dim // 4, 1)
        x[..., d//8:] = shift(x[..., d//8:], self.hidden_dim // 8, 2)
        x = self.mid_ln(F.relu(self.in_proj(x)))
        q, k, v, p = torch.split(x, [
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim * self.expansion_factor
                                   ], -1)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b i h d", h=self.heads), (q, k, v))
        a = contract("b i h d, b j h d -> b h i j", q, k) * (self.head_dim ** -0.5)
        a = self.causal_bias + self.alibi(a)
        a = F.softmax(a, dim=-1)
        o = contract("b h i j, b j h d -> b i h d", a, v)
        o = rearrange(o, "b i h d -> b i (h d)")
        p = contract("b i d, b i d -> b i d", F.gelu(p), torch.roll(p, 1, dims=-1))
        x = torch.cat([o, p], dim=-1)
        x = self.out_proj(x)
        x = self.out_ln(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for the model as a whole. Rather than
        standard residual connections, in this design, each layer aggregates
        the outputs of previous layers gated by learned scalars. These scalars
        are initialized to 1, which makes the default behavior the same as
        normal residual connections, but allows the net to gate-off certain
        paths, & dedicate layer+channel combos to logit output &c.
        """
        super(Transformer, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.sos_index = config.vocab_size
        hidden_dim = config.heads * config.head_dim
        
        embedding = nn.Embedding(config.vocab_size + 1, hidden_dim)
        unembedding = nn.Sequential(*[
                                  nn.LayerNorm((hidden_dim)),
                                  nn.Linear(hidden_dim, config.vocab_size, bias=True),
                                  ])
        self.layers = nn.ModuleList([embedding] + [Block(config) for _ in range(config.depth)] + [unembedding])

        self.gates = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim, i)) for i in range(len(self.layers))])

    def forward(self, x):
        b, i = x.shape
        outputs = []
        for (i, layer) in enumerate(self.layers):
            if i == 0: # Embedding Layer uses original input
              x_in = x
            else: # Non-embedding layers aggregate from previous layers; initialized as residual
              x_in = contract("dl, bidl -> bid", self.gates[i], torch.stack(outputs, -1))
            outputs += [layer(x_in)]
        return outputs[-1]
