"""Observation-conditioned temporal U-Net for action diffusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .nn import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d


class ConditionalResidualTemporalBlock(nn.Module):
    """Modulate temporal features with condition-dependent channel scales and biases."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size=5)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size=5)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels))
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
        h = scale * self.conv1(x) + bias
        return self.conv2(h) + self.skip(x)


class _ConditionalStage(nn.Module):
    """Apply two conditioned residual blocks at one temporal resolution."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.first = ConditionalResidualTemporalBlock(in_channels, out_channels, condition_dim)
        self.second = ConditionalResidualTemporalBlock(out_channels, out_channels, condition_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x, condition), condition)


class ConditionalTemporalUnet(nn.Module):
    """Predict action noise using observation history and diffusion timestep conditions.

    Inputs are actions ``(B, C, N_a)``, timesteps ``(B,)``, and flattened
    observation histories ``(B, N_s)``. Output has the same shape as actions.
    FiLM conditioning follows the observation-global-conditioning approach of
    Chi et al. (2023), "Diffusion Policy: Visuomotor Policy Learning via Action
    Diffusion". Temporal padding supports short and odd action chunk lengths.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        *,
        down_dims: tuple[int, ...] = (128, 256, 512),
        time_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if not down_dims or any(width <= 0 for width in down_dims):
            raise ValueError("down_dims must contain positive channel widths")
        if time_embed_dim < 2 or time_embed_dim % 2:
            raise ValueError("time_embed_dim must be a positive even integer of at least 2")
        self.down_dims = down_dims
        self.time_embed_dim = time_embed_dim
        self.time_embedding = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        condition_dim = state_dim + time_embed_dim
        self.downs = nn.ModuleList(
            _ConditionalStage(in_width, out_width, condition_dim)
            for in_width, out_width in zip((action_dim, *down_dims[:-1]), down_dims, strict=True)
        )
        self.downsamples = nn.ModuleList(Downsample1d(width) for width in down_dims[:-1])
        self.mid = _ConditionalStage(down_dims[-1], down_dims[-1], condition_dim)
        self.upsamples = nn.ModuleList(Upsample1d(width) for width in reversed(down_dims[1:]))
        self.ups = nn.ModuleList(
            _ConditionalStage(deep_width + skip_width, skip_width, condition_dim)
            for deep_width, skip_width in zip(
                reversed(down_dims[1:]), reversed(down_dims[:-1]), strict=True
            )
        )
        self.output = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=5),
            nn.Conv1d(down_dims[0], action_dim, kernel_size=1),
        )

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        length = actions.shape[1]
        x = actions.transpose(1, 2)
        divisor = 2 ** len(self.downsamples)
        padding = (-length) % divisor
        if padding:
            x = F.pad(x, (0, padding))
        time_features = self.time_mlp(self.time_embedding(timesteps).to(dtype=actions.dtype))
        condition = torch.cat((time_features, state), dim=-1)
        skips = []
        for stage, downsample in zip(self.downs[:-1], self.downsamples, strict=True):
            x = stage(x, condition)
            skips.append(x)
            x = downsample(x)
        x = self.mid(self.downs[-1](x, condition), condition)
        for upsample, stage, skip in zip(self.upsamples, self.ups, reversed(skips), strict=True):
            x = upsample(x)
            x = stage(torch.cat((x, skip), dim=1), condition)
        return self.output(x)[..., :length].transpose(1, 2)
