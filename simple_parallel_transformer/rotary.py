import torch
import torch.nn as nn

from torch import Tensor

class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
        max_tsz: int,
        embed_dim: int,
        learnable: bool = False,
        base: int = 10_000,
    ) -> None:
        """Defines a rotary embeddings module.

        Args:
            max_tsz: The maximum sequence length.
            embed_dim: The embedding dimension.
            learnable: Whether the embeddings are learnable.
            base: The base for the sinusoidal embeddings.
        """
        super().__init__()

        assert embed_dim % 4 == 0, "Embedding dimension must be divisible by 4."

        self.embed_dim = embed_dim
        self.learnable = learnable
        self.base = base

        cos, sin = self.get_embeddings(max_tsz)
        self.cos, seflf.sin = nn.Parameter(cos, requires_grad=learnable), nn.Parameter(sin, requires_grad=learnable)

    def get_embeddings(self, tsz: int) -> tuple[Tensor, Tensor]:
        half_d = self.embed_dim // 2
        theta = 1.0 / (self.base ** (torch.arange(0, half_d, 2).float() / half_d))
        seq_idx = torch.arange(tsz).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        return idx_theta2.cos(), idx_theta2.sin()

    def _neg_half(self, x: Tensor) -> Tensor:
        quarter_d = self.embed_dim // 4
        return torch.cat([-x[..., quarter_d:], x[..., :quarter_d]], dim=-1)

    def forward(self, x: Tensor, offset: int = 0, times: Tensor | None = None) -> Tensor:
        half_d = self.embed_dim // 2
        x_rope, x_pass = x[..., :half_d], x[..., half_d:]
        neg_half_x = self._neg_half(x_rope)
        cos_part = self.cos[None, offset : offset + x.shape[1]] if times is None else self.cos[times]
        sin_part = self.sin[None, offset : offset + x.shape[1]] if times is None else self.sin[times]
        x_rope = x_rope * cos_part + neg_half_x * sin_part
        return torch.cat((x_rope, x_pass), dim=-1)
