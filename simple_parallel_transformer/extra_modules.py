import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
from opt_einsum import contract

from dataclasses import dataclass
from torch_discounted_cumsum import discounted_cumsum_left

def exists(val):
    return val is not None

class SoftPrefixMax(nn.Module):
    def __init__(self, dimensions):
        super(SoftPrefixMax, self).__init__()
        self.dimensions = dimensions
    
    def forward(self, x):
        part = x[..., :self.dimensions]
        x[..., :self.dimensions] = torch.logcumsumexp(part * 5.0, dim=1) / 5.0
        return x
        
class CPProjection(nn.Module):
    def __init__(self, input_dimension, output_dimension, bond_dim=1):
        super(CPProjection, self).__init__()
        assert output_dimension % 64 == 0
        factors = [8, 8, output_dimension // 64]
        self.bond_dim = bond_dim
        self.A = nn.Linear(input_dimension, factors[0] * bond_dim, bias=False)
        self.B = nn.Linear(input_dimension, factors[1] * bond_dim, bias=False)
        self.C = nn.Linear(input_dimension, factors[2] * bond_dim, bias=False)
    
    def forward(self, x):
      a = self.A(x)
      b = self.B(x)
      c = self.C(x)
      a = rearrange(a, "b i (z d) -> b i z d", d=self.bond_dim)
      b = rearrange(b, "b i (o d) -> b i o d", d=self.bond_dim)
      c = rearrange(c, "b i (t d) -> b i t d", d=self.bond_dim)
      out = contract("bizd, biod, bitd -> bizot", a, b, c)
      out = rearrange(out, "b i z o t -> b i (z o t)")
      return out
    
class TensorTrainProjection(nn.Module):
    def __init__(self, input_dimension, output_multiplier=1, rank=1, init='normal'):
        super(TensorTrainProjection, self).__init__()
        assert input_dimension % 512 == 0
        input_multiplier = input_dimension // 512
        top_ranks = [2 * input_multiplier] + [2]*8
        bottom_ranks = [2 * output_multiplier] + [2]*8
        left_ranks = []
        right_ranks = []
        for i, (top_rank, bottom_rank) in enumerate(zip(top_ranks, bottom_ranks)):
          if i == 0:
            left_ranks += [1]
          else:
            left_ranks += [rank]

          if i + 1 == 9:
            right_ranks += [1]
          else:
            right_ranks += [rank]

        self.top_ranks = top_ranks
        self.bottom_ranks = bottom_ranks
        self.left_ranks = left_ranks
        self.right_ranks = right_ranks

        print({
          'top': top_ranks,
          'bottom': bottom_ranks,
          'left': left_ranks,
          'right': right_ranks,
        })

        print(f"Top rank product {product(top_ranks)}")
        print(f"Bottom rank product {product(bottom_ranks)}")

        self.projections = nn.ParameterList([
          nn.Parameter(torch.randn((top_rank, bottom_rank, left_rank, right_rank))) for (top_rank, bottom_rank, left_rank, right_rank) in zip(
            self.top_ranks,
            self.bottom_ranks,
            self.left_ranks,
            self.right_ranks
          )
        ])

        if init == 'zeros':
          for i in range(len(self.projections)):
            nn.init.zeros_(self.projections[i])

        print(self.projections)

    def forward(self, x):
        x = rearrange(x, "... (a b c d e f g h i) -> ... a b c d e f g h i",
          a=self.top_ranks[0],
          b=self.top_ranks[1],
          c=self.top_ranks[2],
          d=self.top_ranks[3],
          e=self.top_ranks[4],
          f=self.top_ranks[5],
          g=self.top_ranks[6],
          h=self.top_ranks[7],
          i=self.top_ranks[8])

          # o-o-o-o-o-o-o-o-o

        p = self.projections

        # tops:    a b c d e f g h i
        # bonds:    s t u v w x y z
        # bottoms: j k l m n o p q r

        output = contract("... abcdefghi, ajs, bkst, cltu, dmuv, envw, fowx, gpxy, hqzy, irz -> ... jklmnopqr", 
          x,
          p[0][:, :, 0, :], 
          p[1], 
          p[2], 
          p[3], 
          p[4], 
          p[5], 
          p[6], 
          p[7], 
          p[8][:, :, :, 0])
        
        output = rearrange(output, "... j k l m n o p q r -> ... (j k l m n o p q r)")
        return output
 
class KroneckerProjection(nn.Module):
    def __init__(self, input_dimension, multiplier):
        super(KroneckerProjection, self).__init__()
        # 512 -> 16 x 32
        # 2048 -> 32 x 64
        # A = 16 x 32
        # B = 32 x 64
        # C = (16 * 32) x (32 * 64) = 512 x 2048
        output_dimension = input_dimension * multiplier
        assert input_dimension % 16 == 0
        assert output_dimension % 16 == 0
        a_factors = [input_dimension // 16, 16]
        b_factors = [16, output_dimension // 16]
        self.A = nn.Parameter(torch.randn(a_factors[0], a_factors[1]))
        self.B = nn.Parameter(torch.randn(b_factors[0], b_factors[1]))

    def forward(self, x):
        C = torch.kron(self.A, self.B)
        out = einsum("... m, m n -> ... n", x, C)
        return out
    
class LRU(nn.Module):
    def __init__(self, d_hidden, r_min=0.4, r_max=0.99):
        super(LRU, self).__init__()
        uniforms = torch.rand(d_hidden)
        r_min_squared = r_min ** 2
        r_max_squared = r_max ** 2
        nus = - torch.log(uniforms * (r_max_squared - r_min_squared) + r_min_squared) / 2.
        nu_logs = torch.log(nus)
        self.nu_logs = nn.Parameter(nu_logs)

    def forward(self, x):
      nus = torch.exp(self.nu_logs)
      lambdas = torch.exp(-nus)
      gammas = (1. - (torch.abs(lambdas) ** 2)) ** 0.5
      values = gammas[None, None] * x
      b, i, d = x.shape
      
      batch_lambdas = repeat(lambdas, "d -> (b d)", b=b)
      values = rearrange(values, "b i d -> (b d) i")
      hidden = discounted_cumsum_left(values, batch_lambdas)
      hidden = rearrange(hidden, "(b d) i -> b i d", b=b, d=d)
      return hidden
    
