"""Neural-network helpers for diffusion/value models.

Implements the temporal U-Net architecture from Janner et al. (2022) "Planning
with Diffusion for Flexible Behavior Synthesis".  The core idea is to use 1D
temporal convolutions with a small receptive field so each denoising step only
enforces local consistency; composing many steps drives global coherence.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# Time-step embedding
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps (Vaswani et al.)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] integer timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → SiLU building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        # GroupNorm requires n_groups to divide out_channels.
        actual_groups = max(1, min(n_groups, out_channels))
        while out_channels % actual_groups != 0 and actual_groups > 1:
            actual_groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(actual_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """Two Conv1dBlocks with a time-step conditioning and skip connection.

    Input ``x``: ``[B, in_channels, H]`` (channels-first, as used by Conv1d).
    Input ``t``: ``[B, embed_dim]`` time embedding.
    Output: ``[B, out_channels, H]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size)
        # Projects time embedding → out_channels, broadcast over H.
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )
        # 1x1 projection for skip connection when channel sizes differ.
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H],  t: [B, embed_dim]
        h = self.conv1(x) + self.time_proj(t).unsqueeze(-1)  # broadcast over H
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample1d(nn.Module):
    """Halve the temporal resolution with a strided convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Double the temporal resolution with a transposed convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Temporal U-Net (diffusion denoiser)
# ---------------------------------------------------------------------------


class TemporalUnet(nn.Module):
    """Temporal U-Net for trajectory denoising.

    Architecture from Janner et al. 2022 (Figure 2 / Appendix A):
    - 1D temporal convolutions with a small receptive field (kernel_size=5)
    - Multi-scale encoder / decoder with skip connections
    - Sinusoidal time-step conditioning injected into every residual block

    Input shape:  ``x: [B, H, transition_dim]``, ``t: [B]`` (integer timesteps)
    Output shape: ``[B, H, transition_dim]`` (predicted noise ε)
    """

    def __init__(
        self,
        transition_dim: int,
        *,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if dim % 8 != 0:
            raise ValueError(f"dim must be divisible by 8 for GroupNorm, got {dim}")

        self.transition_dim = int(transition_dim)
        self.dim = int(dim)
        self.dim_mults = tuple(dim_mults)
        self._n_downs = len(dim_mults) - 1  # number of Downsample/Upsample ops

        time_embed_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Channel sizes at each U-Net level.
        dims = [transition_dim] + [dim * m for m in dim_mults]
        in_out = list(pairwise(dims))

        self.downs = nn.ModuleList([])
        for idx, (d_in, d_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(d_in, d_out, time_embed_dim, kernel_size),
                        ResidualTemporalBlock(d_out, d_out, time_embed_dim, kernel_size),
                        nn.Identity() if is_last else Downsample1d(d_out),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, time_embed_dim, kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, time_embed_dim, kernel_size)

        self.ups = nn.ModuleList([])
        for idx, (d_in, d_out) in enumerate(list(reversed(in_out[1:]))):
            is_last = idx >= len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        # skip connection doubles the input channels
                        ResidualTemporalBlock(d_out * 2, d_in, time_embed_dim, kernel_size),
                        ResidualTemporalBlock(d_in, d_in, time_embed_dim, kernel_size),
                        nn.Identity() if is_last else Upsample1d(d_in),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(dims[1], dims[1], kernel_size),
            nn.Conv1d(dims[1], transition_dim, 1),
        )

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad sequence length to the nearest multiple of 2^n_downs."""
        if self._n_downs == 0:
            return x, 0
        divisor = 2**self._n_downs
        h = x.shape[-1]
        pad = (-h) % divisor  # smallest non-negative pad to reach a multiple
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x, pad

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [batch, horizon, dim],  t: [batch]
        orig_h = x.shape[1]
        x = x.permute(0, 2, 1)  # [batch, dim, horizon] - channels-first for Conv1d
        t = t.to(device=x.device)

        x, pad = self._pad_to_multiple(x)

        t_emb = self.time_mlp(t)  # [B, time_embed_dim]

        skips: list[torch.Tensor] = []
        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, t_emb)
            x = resnet2(x, t_emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        for resnet1, resnet2, upsample in self.ups:
            skip = skips.pop()
            # Crop skip to match x if sizes differ (can happen with odd lengths).
            if skip.shape[-1] != x.shape[-1]:
                skip = skip[..., : x.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = resnet1(x, t_emb)
            x = resnet2(x, t_emb)
            x = upsample(x)

        x = self.final_conv(x)

        # Crop padding, restore original shape.
        if pad > 0:
            x = x[..., :orig_h]
        return x.permute(0, 2, 1)  # [batch, horizon, dim]


# ---------------------------------------------------------------------------
# Temporal value network (encoder-only U-Net → scalar)
# ---------------------------------------------------------------------------


class TemporalValueNet(nn.Module):
    """Trajectory value estimator based on the temporal U-Net encoder.

    Encodes a trajectory ``[B, H, D]`` with a U-Net downsampling path, then
    global-averages over the time dimension and projects to a scalar ``[B, 1]``.
    Used as J_φ in the guided diffusion planning algorithm.

    This model is differentiable w.r.t. its input, enabling gradient-based
    guidance during reverse diffusion sampling.
    """

    def __init__(
        self,
        transition_dim: int,
        *,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if dim % 8 != 0:
            raise ValueError(f"dim must be divisible by 8 for GroupNorm, got {dim}")

        self.transition_dim = int(transition_dim)
        self.dim = int(dim)
        self.dim_mults = tuple(dim_mults)
        self._n_downs = len(dim_mults) - 1

        time_embed_dim = dim * 4  # unused at inference but kept for architectural parity
        # Lightweight time MLP (identity by default - value model usually not time-conditioned).
        self._time_embed_dim = time_embed_dim

        dims = [transition_dim] + [dim * m for m in dim_mults]
        in_out = list(pairwise(dims))

        self.encoder = nn.ModuleList([])
        for idx, (d_in, d_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            self.encoder.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(d_in, d_out, time_embed_dim, kernel_size),
                        ResidualTemporalBlock(d_out, d_out, time_embed_dim, kernel_size),
                        nn.Identity() if is_last else Downsample1d(d_out),
                    ]
                )
            )

        # Dummy constant time embedding (value model does not use diffusion timestep).
        mid_dim = dims[-1]
        self.mid_block = ResidualTemporalBlock(mid_dim, mid_dim, time_embed_dim, kernel_size)
        # Head: global average pool over H, then linear to scalar.
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mid_dim, 1),
        )

        # Register a learned constant to fill the time-embedding slot.
        self.register_buffer("_const_t", torch.zeros(time_embed_dim))

    def _get_t_emb(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._const_t.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self._n_downs == 0:
            return x, 0
        divisor = 2**self._n_downs
        h = x.shape[-1]
        pad = (-h) % divisor
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x, pad

    def forward(self, x: torch.Tensor, condition: object = None) -> torch.Tensor:
        # x: [batch, horizon, dim]; condition accepted but ignored (use inpainting instead).
        batch = x.shape[0]
        x = x.permute(0, 2, 1)  # [batch, dim, horizon]

        x, _ = self._pad_to_multiple(x)
        t_emb = self._get_t_emb(batch, device=x.device, dtype=x.dtype)

        for resnet1, resnet2, downsample in self.encoder:
            x = resnet1(x, t_emb)
            x = resnet2(x, t_emb)
            x = downsample(x)

        x = self.mid_block(x, t_emb)
        # Global average over the (now downsampled) time dimension.
        x = x.mean(dim=-1)  # [B, mid_dim]
        return self.head(x)  # [B, 1]


__all__ = [
    "Conv1dBlock",
    "Downsample1d",
    "ResidualTemporalBlock",
    "SinusoidalPosEmb",
    "TemporalUnet",
    "TemporalValueNet",
    "Upsample1d",
]
