from einops import rearrange
from opt_einsum import contract
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from torch import zeros, ones, exp, linspace, triu, split, normal
from torch.nn import Module, Embedding, Linear, LayerNorm, ModuleList, Sequential, Parameter
from torch.nn.init import normal_
from torch.nn.functional import pad, sigmoid, silu, scaled_dot_product_attention


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
cs.store(name="config", node=Config)


class FanoutLinear(Module):
    def __init__(self, d_in, d_outs):
        super(FanoutLinear, self).__init__()
        self.d_in, self.d_outs = d_in, d_outs
        self.merged = Linear(d_in, sum(self.d_outs), bias=True)

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
        return split(self.merged(x), self.d_outs, dim=-1)


class Block(Module):
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.heads, self.d_model = config.heads, config.d_model
        self.d_expanded = self.d_model * config.expansion_factor
        self.d_head = (self.d_expanded) // self.heads
        self.in_ln, self.out_ln = LayerNorm(self.d_model), LayerNorm(self.d_model)
        self.in_proj = FanoutLinear(self.d_model, 4 * [self.d_expanded] + 2 * [self.heads])
        self.in_proj.init(4, weight=zeros((self.heads, self.d_model)), bias=linspace(-6., 6., self.heads))
        self.in_proj.init(5, weight=zeros((self.heads, self.d_model)), bias=linspace(-6., 6., self.heads))
        self.log_scale = Parameter(zeros(self.heads))
        self.out_proj = Linear(self.d_expanded, self.d_model, bias=False)
        normal_(self.out_proj.weight, mean=0.0, std=0.02 * ((2 * config.depth) ** -0.5))
        
    def forward(self, x):
        (b, i, d), device = (x.shape, x.device)
        q, k, v, p, smear, dpos = self.in_proj(self.in_ln(x))
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b h i d", h=self.heads), (q, k, v))
        smear = rearrange(sigmoid(smear), "b j h -> b h j")[..., None]
        k = ((1 - smear) * k) + (smear * pad(k, (0, 0, 1, -1)))
        scale = exp(self.log_scale)[None, :, None, None]
        pos = sigmoid(rearrange(dpos, "b i h -> b h i")).cumsum(dim=-1)
        relpos = pos[..., None] - pos[..., None, :]
        mask = triu(-1e10 * ones((i, i), device=device), diagonal=1) - relpos
        o = scaled_dot_product_attention(q / scale, k / scale, v, attn_mask=mask)
        o = rearrange(o, "b h i d -> b i (h d)")
        return self.out_ln(self.out_proj(silu(p) * o))


class Transformer(Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.max_seq_len, self.sos_index = config.max_seq_len, config.vocab_size
        self.embed = Embedding(config.vocab_size + 1, config.d_model)
        self.blocks = ModuleList([Block(config) for i in range(config.depth)])
        self.unembed = Sequential(*[
                                  LayerNorm((config.d_model)),
                                  Linear(config.d_model, config.vocab_size, bias=True),
                                  ])

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = x + block(x)
        return self.unembed(x)
