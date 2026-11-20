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

from mujoco_lab.learning.infrastructure.utils import build_mlp

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
    """Two Conv1dBlocks with additive or scale/bias conditioning and a skip.

    Input ``x``: ``[B, in_channels, H]`` (channels-first, as used by Conv1d).
    Input ``condition``: ``[B, embed_dim]`` conditioning features.
    Output: ``[B, out_channels, H]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: int = 5,
        *,
        scale_bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size)
        self.scale_bias = scale_bias
        if scale_bias:
            self.condition = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * out_channels))
        else:
            # Keep the value encoder's additive conditioning and parameter names.
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                *build_mlp(input_dim=embed_dim, output_dim=out_channels, hidden_layers=()),
            )
        # 1x1 projection for skip connection when channel sizes differ.
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if self.scale_bias:
            scale, bias = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
            h = scale * self.conv1(x) + bias
        else:
            h = self.conv1(x) + self.time_proj(condition).unsqueeze(-1)
        return self.conv2(h) + self.skip(x)


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


class _TemporalStage(nn.Module):
    """Apply two FiLM residual blocks at one temporal resolution."""

    def __init__(
        self, in_channels: int, out_channels: int, condition_dim: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.first = ResidualTemporalBlock(
            in_channels, out_channels, condition_dim, kernel_size, scale_bias=True
        )
        self.second = ResidualTemporalBlock(
            out_channels, out_channels, condition_dim, kernel_size, scale_bias=True
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x, condition), condition)


class TemporalUnet(nn.Module):
    """Denoise temporal sequences with an optional global condition.

    Temporal convolutions follow Janner et al. (2022), "Planning with Diffusion
    for Flexible Behavior Synthesis". FiLM conditioning follows Chi et al.
    (2023), "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion".

    Inputs are sequences ``(B, H, transition_dim)``, timesteps ``(B,)``, and an
    optional condition ``(B, condition_dim)``. Output matches the sequence shape.
    ``down_dims`` sets each level's channel width; ``time_embed_dim`` sets the
    timestep feature width independently. With ``condition_dim=0``, only the
    timestep conditions the network. An omitted condition uses zeros for its
    configured features. Temporal padding supports short and odd horizons.
    """

    def __init__(
        self,
        transition_dim: int,
        *,
        down_dims: tuple[int, ...] = (32, 64, 128, 256),
        time_embed_dim: int = 128,
        condition_dim: int = 0,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if not down_dims or any(width <= 0 for width in down_dims):
            raise ValueError("down_dims must contain positive channel widths")
        if time_embed_dim < 2 or time_embed_dim % 2:
            raise ValueError("time_embed_dim must be a positive even integer of at least 2")
        if condition_dim < 0:
            raise ValueError("condition_dim must be nonnegative")
        self.transition_dim = transition_dim
        self.down_dims = down_dims
        self.time_embed_dim = time_embed_dim
        self.condition_dim = condition_dim
        self.time_embedding = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        embed_dim = time_embed_dim + condition_dim
        self.downs = nn.ModuleList(
            _TemporalStage(in_width, out_width, embed_dim, kernel_size)
            for in_width, out_width in zip(
                (transition_dim, *down_dims[:-1]), down_dims, strict=True
            )
        )
        self.downsamples = nn.ModuleList(Downsample1d(width) for width in down_dims[:-1])
        self.mid = _TemporalStage(down_dims[-1], down_dims[-1], embed_dim, kernel_size)
        self.upsamples = nn.ModuleList(Upsample1d(width) for width in reversed(down_dims[1:]))
        self.ups = nn.ModuleList(
            _TemporalStage(deep_width + skip_width, skip_width, embed_dim, kernel_size)
            for deep_width, skip_width in zip(
                reversed(down_dims[1:]), reversed(down_dims[:-1]), strict=True
            )
        )
        self.output = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], transition_dim, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is None:
            if self.condition_dim:
                condition = x.new_zeros((x.shape[0], self.condition_dim))
        elif condition.shape != (x.shape[0], self.condition_dim):
            raise ValueError("condition must have shape (B, condition_dim)")
        length = x.shape[1]
        x = x.transpose(1, 2)
        divisor = 2 ** len(self.downsamples)
        padding = (-length) % divisor
        if padding:
            x = F.pad(x, (0, padding))
        time_features = self.time_mlp(
            self.time_embedding(timesteps.to(device=x.device)).to(dtype=x.dtype)
        )
        features = (
            time_features if condition is None else torch.cat((time_features, condition), dim=-1)
        )
        skips = []
        for stage, downsample in zip(self.downs[:-1], self.downsamples, strict=True):
            x = stage(x, features)
            skips.append(x)
            x = downsample(x)
        x = self.mid(self.downs[-1](x, features), features)
        for upsample, stage, skip in zip(self.upsamples, self.ups, reversed(skips), strict=True):
            x = upsample(x)
            x = stage(torch.cat((x, skip), dim=1), features)
        return self.output(x)[..., :length].transpose(1, 2)


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
            *build_mlp(input_dim=mid_dim, output_dim=1, hidden_layers=()),
        )

        # Register a learned constant to fill the time-embedding slot.
        self.register_buffer("_const_t", torch.zeros(time_embed_dim))

    _const_t: torch.Tensor

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, horizon, dim].
        batch = x.shape[0]
        x = x.permute(0, 2, 1)  # [batch, dim, horizon]

        x, _ = self._pad_to_multiple(x)
        t_emb = self._get_t_emb(batch, device=x.device, dtype=x.dtype)

        # __init__ creates ModuleList stages; PyTorch types their elements as Module.
        for resnet1, resnet2, downsample in self.encoder:  # ty: ignore[not-iterable]
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
