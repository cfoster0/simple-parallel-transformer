import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from opt_einsum import contract
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class Config:
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


class FanoutLinear(nn.Module):
    def __init__(self, d_in, d_outs):
        super(FanoutLinear, self).__init__()
        self.d_in, self.d_outs = d_in, d_outs
        self.merged = nn.Linear(d_in, sum(self.d_outs), bias=True)

    def init(self, index, weight, bias=None):
        start_idx, idx_span = sum(self.d_outs[:index]), self.d_outs[index]
        if (self.d_outs[index], self.d_in) != weight.shape:
            raise Exception(f"`Expected shape {(self.d_outs[index], self.d_in)} but got {weight.shape}.")
        self.merged.weight.data[start_idx:start_idx+idx_span, :] = weight
        if bias is not None:
          if (self.d_outs[index],) != bias.shape:
              raise Exception(f"`Expected shape {(self.d_outs[index],)} but got {bias.shape}.")
          self.merged.bias.data[start_idx:start_idx+idx_span] = bias
        
    def forward(self, x):
        return torch.split(self.merged(x), self.d_outs, dim=-1)


class Block(nn.Module):
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.heads, self.d_model = config.heads, config.d_model
        self.d_expanded = self.d_model * config.expansion_factor
        self.d_head = (self.d_expanded) // self.heads
        self.in_ln = nn.LayerNorm(self.d_model)
        self.out_ln = nn.LayerNorm(self.d_model)
        self.in_proj = FanoutLinear(self.d_model, 4 * [self.d_expanded] + [self.heads])
        position_weight_init = torch.zeros((self.heads, self.d_model))
        position_bias_init = torch.linspace(-8., 8., steps=self.heads)
        self.in_proj.init(4, weight=position_weight_init, bias=position_bias_init)
        self.out_proj = nn.Linear(self.d_expanded, self.d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 * ((2 * config.depth) ** -0.5))
        self.smear_factor = nn.Parameter(torch.linspace(-6., 6., self.heads))
        self.log_scale = nn.Parameter(torch.zeros(self.heads))
        
    def forward(self, x):
        (b, i, d), device = (x.shape, x.device)
        q, k, v, p, y = self.in_proj(self.in_ln(x))
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b h i d", h=self.heads), (q, k, v))
        smear = F.sigmoid(self.smear_factor)[None, :, None, None]
        k = ((1 - smear) * k) + (smear * F.pad(k, (0, 0, 1, -1)))
        s = torch.exp(self.log_scale)[None, :, None, None]
        pos = F.sigmoid(rearrange(y, "b i h -> b h i")).cumsum(dim=-1)
        relpos = pos[..., None] - pos[..., None, :]
        mask = torch.triu(-1e10 * torch.ones((i, i), device=device), diagonal=1) - relpos
        o = F.scaled_dot_product_attention(q / s, k / s, v, attn_mask=mask)
        o = rearrange(o, "b h i d -> b i (h d)")
        return self.out_ln(self.out_proj(F.silu(p) * o))


class Transformer(nn.Module):
    def __init__(self, config: Config):
        """
        In this module is the code for the model as a whole.
        """
        super(Transformer, self).__init__()
        self.max_seq_len, self.sos_index = config.max_seq_len, config.vocab_size
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
