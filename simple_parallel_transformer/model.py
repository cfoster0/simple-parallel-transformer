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
    d_head: int
    vocab_size: int
    max_seq_len: int = 2048
    expansion_factor: int = 4
    seed: int = 10101
        
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)

# N-timestep delay, assuming time is penultimate axis
def shift(x, n):
  return F.pad(x, (0, 0, n, -n), value=0)

class LearnedALiBi(nn.Module):
    def __init__(self, heads):
        super().__init__()
        slopes = torch.cat([torch.logspace(0., -6., steps=heads//2, base=2.), torch.zeros(heads - (heads//2))], dim=0)
        self.slopes = nn.Parameter(rearrange(slopes, 'h -> () h () ()'))
        self.register_buffer('bias', None, persistent = False)

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if (not self.training) and (self.bias is not None) and (self.bias.shape[-1] >= j):
            return qk_dots + self.bias[..., :j]

        bias = rearrange(torch.arange(j, device = device), 'j -> () () () j') * self.slopes
        bias = F.pad(bias, (0, 0, 0, 0, 0, h - bias.shape[1]))
        self.register_buffer('bias', bias, persistent = False)
        return qk_dots + bias
    
class Block(nn.Module):
    def __init__(self, config: Config, depth: int):
        """
        In this module is the code for a transformer-like block. It builds on
        several new ideas since the original "Attention is all you need":
        (1) Cogview's Sandwich Layer Norm, which puts a LayerNorm at the start 
            and end of the block
        (2) Parallel attention and MLP, originated by Ben Wang
        (3) Token shift--originated by BlinkDL--in the first 2 layers on 
            a portion of dimensions to provide direct access to
            previous token representations & for easy n-gram learning
        (4) a home-grown modification that adds a LayerNorm after the 
            initial projection & token shift, similar in spirit to Normformer
        (5) learned per-head linear biases on the attention logits similar 
            to ALiBi; half of heads are initialized as nonspatial
        (6) a bilinear gated linear unit (GLU) to replace the traditional 
            ReLu/GELU nonlinearity in the MLP

        ---
        References:
        Sandwich Layer Norm - https://arxiv.org/abs/2105.13290
        Parallel Block - https://github.com/kingoflolz/mesh-transformer-jax
        Token Shift - https://github.com/BlinkDL/RWKV-LM
        Normformer - https://arxiv.org/abs/2110.09456
        ALiBi - https://arxiv.org/abs/2108.12409
        Bilinear GLU - https://arxiv.org/abs/2002.05202v1
        """
        super(Block, self).__init__()
        self.heads = config.heads
        self.d_head = config.d_head
        self.max_seq_len = config.max_seq_len
        self.d_model = self.heads * self.d_head
        self.d_bilinear = (self.d_model * config.expansion_factor) // 2
        self.depth = depth
        
        self.in_ln = nn.LayerNorm(self.d_model)
        self.mid_ln = nn.LayerNorm((self.d_model * 3) + (self.d_bilinear * 2))
        self.out_ln = nn.LayerNorm(self.d_model)

        self.in_proj = nn.Linear(self.d_model, (self.d_model * 3) + (self.d_bilinear * 2), bias=False)
        self.out_proj = nn.Linear(self.d_model + self.d_bilinear, self.d_model, bias=False)
        nn.init.orthogonal_(self.in_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

        causal_mask = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len)))
        self.register_buffer('causal_bias', rearrange(-1e10 * (1. - causal_mask), "i j -> () () i j"))
        self.alibi = LearnedALiBi(self.heads)
        
    def forward(self, x):
        b, l, d = x.shape

        x = self.in_ln(x)
        if self.depth < 2:
          x[..., :d//4] = shift(x[..., :d//4], 1)
          x[..., -d//8:] = shift(x[..., -d//8:], 2)
        q, k, v, p1, p2 = torch.split(self.mid_ln(self.in_proj(x)), [
                                   self.d_model,
                                   self.d_model,
                                   self.d_model,
                                   self.d_bilinear,
                                   self.d_bilinear,
                                   ], -1)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b h i d", h=self.heads), (q, k, v))
        logits = contract("b h i d, b h j d -> b h i j", q, k) * (self.d_head ** -0.5)
        a = F.softmax(self.causal_bias[..., :l, :l] + self.alibi(logits), dim=-1)
        o = rearrange(contract("b h i j, b h j d -> b h i d", a, v), "b h i d -> b i (h d)")
        x = self.out_ln(self.out_proj(torch.cat([o, p1 * p2], dim=-1)))
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
        d_model = config.heads * config.d_head
        
        embedding = nn.Embedding(config.vocab_size + 1, d_model)
        unembedding = nn.Sequential(*[
                                  nn.LayerNorm((d_model)),
                                  nn.Linear(d_model, config.vocab_size, bias=True),
                                  ])
        self.layers = nn.ModuleList([embedding] + [Block(config, i) for i in range(config.depth)] + [unembedding])

        self.gates = nn.ParameterList([nn.Parameter(torch.ones(d_model, i)) for i in range(len(self.layers))])

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
