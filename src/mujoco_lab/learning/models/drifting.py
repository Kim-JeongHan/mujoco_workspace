"""
Reference: lambertae/drifting, drift_loss.py
Paper: Deng et al., Generative Modeling via Drifting, arXiv:2602.04770.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["drift_loss", "DriftingLoss"]


# Formula reference: https://arxiv.org/html/2602.04770v1.
def _prepare_weight(value: Tensor | None, samples: Tensor) -> Tensor:
    if value is None:
        return torch.ones(samples.shape[:2], device=samples.device, dtype=torch.float32)
    return value.float()


def drift_loss(
    gen: Tensor,
    fixed_pos: Tensor,
    fixed_neg: Tensor | None = None,
    weight_gen: Tensor | None = None,
    weight_pos: Tensor | None = None,
    weight_neg: Tensor | None = None,
    R_list: Sequence[float] = (0.02, 0.05, 0.2),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the upstream drifting loss in float32.

    Args:
        gen: [B, G, D], generated samples WITH their autograd graph.
        fixed_pos: [B, P, D], reference/positive samples, P >= 1.
        fixed_neg: Optional [B, N, D] extra negative samples. Generated
            samples already supply repulsion; this argument is not required.
        weight_gen, weight_pos, weight_neg: Optional nonnegative target
            weights [B, G], [B, P], [B, N]. These are not query/loss weights.
        R_list: Positive kernel temperatures, not sampling timesteps.

    Returns:
        loss_per_group: [B], float32. Use .mean() before backward().
        info: Detached scalar tensors: scale and loss_<R>. loss_<R> is the
            raw force mean-square BEFORE force normalization, not a separate
            regression loss. All scale statistics are global over this call.

    Each B entry is a separate distribution/condition group. For unconditional
    training use B=1 and put all generated samples on G. The distance and
    force normalization statistics are global, as in the original JAX code.
    Gradients flow ONLY through gen / coordinate_scale, not through targets,
    weights, distances, force, or either normalization statistic.
    """
    if fixed_neg is None:
        fixed_neg = gen.new_empty((gen.shape[0], 0, gen.shape[2]))
    temperatures = tuple(R_list)

    # Disabling autocast is important: float() alone does not prevent a
    # surrounding autocast context from downcasting the matrix products.
    with torch.autocast(device_type=gen.device.type, enabled=False):
        prediction = gen.float()  # KEEP the generator's gradient path.
        with torch.no_grad():  # stopgrad on the target construction.
            query = prediction.detach()
            positive = fixed_pos.detach().float()
            negative = fixed_neg.detach().float()
            wg = _prepare_weight(weight_gen, query)
            wp = _prepare_weight(weight_pos, positive)
            wn = _prepare_weight(weight_neg, negative)
            targets = torch.cat((query, negative, positive), dim=1)
            weights = torch.cat((wg, wn, wp), dim=1)

            # Keep the upstream sqrt(clamp(squared_distance, 1e-8)) rather
            # than replacing it with torch.cdist, which has a different floor.
            squared_distance = (
                query.square().sum(dim=-1, keepdim=True)  # ||q||^2: [B, G, 1]
                + targets.square().sum(dim=-1).unsqueeze(1)  # ||t||^2: [B, 1, T]
                - 2.0 * torch.bmm(query, targets.transpose(1, 2))  # -2 q^T t: [B, G, T]
            )  # ||q - t||^2 for every query-target pair: [B, G, T]

            distances = squared_distance.clamp_min(1e-8).sqrt()
            # Weighted distance mean m; S = m / sqrt(D).

            distance_scale = (distances * weights.unsqueeze(1)).mean() / weights.mean()
            coordinate_scale = (distance_scale / math.sqrt(gen.shape[2])).clamp_min(1e-3)
            # normalized coordinates x_tilde = x / S.
            query_scaled = query / coordinate_scale
            targets_scaled = targets / coordinate_scale
            # d / m = ||x_tilde - y_tilde|| / sqrt(D), before floors.
            kernel_distance = distances / distance_scale.clamp_min(1e-3)

            n_generated = gen.shape[1]  # G
            n_negative = n_generated + negative.shape[1]  # N
            # The original uses an additive distance penalty, NOT -inf logits.
            diagonal = torch.eye(n_generated, device=gen.device, dtype=torch.float32)
            # pad the self-comparison mask to cover all targets.
            self_penalty = F.pad(diagonal, (0, negative.shape[1] + positive.shape[1]))
            kernel_distance = kernel_distance + 100.0 * self_penalty.unsqueeze(0)

            drift = torch.zeros_like(query_scaled)
            info: dict[str, Tensor] = {"scale": distance_scale}
            for temperature in temperatures:
                # log k(x, y) = -normalized_distance / tau.
                logits = -kernel_distance / temperature
                row_probability = logits.softmax(dim=-1)
                column_probability = logits.softmax(dim=-2)
                # A_ij = sqrt(softmax_row(log k) * softmax_col(log k)).
                # Retain the 1e-6 floor even for masked/self entries. Explicitly
                # zeroing the diagonal afterwards would change the algorithm.
                affinity = (row_probability * column_probability).clamp_min(1e-6).sqrt()
                affinity = affinity * weights.unsqueeze(1)
                negative_affinity = affinity[:, :, :n_negative]
                positive_affinity = affinity[:, :, n_negative:]

                # c^- = -A^- sum(A^+); c^+ = A^+ sum(A^-).
                negative_coefficient = -negative_affinity * positive_affinity.sum(
                    dim=-1, keepdim=True
                )
                positive_coefficient = positive_affinity * negative_affinity.sum(
                    dim=-1, keepdim=True
                )
                coefficients = torch.cat((negative_coefficient, positive_coefficient), dim=-1)
                # V_i = sum_j c_ij (y_j - x_i), attraction minus repulsion.
                force = torch.bmm(coefficients, targets_scaled)
                force = force - coefficients.sum(dim=-1, keepdim=True) * query_scaled
                # lambda^2 = mean(V^2); V_tilde = V / lambda.
                force_mse = force.square().mean()  # Global, NOT per group.
                info[f"loss_{temperature}"] = force_mse
                # add the normalized fields across temperatures.
                drift = drift + force / force_mse.clamp_min(1e-8).sqrt()

            regression_target = query_scaled + drift  # sg(x_tilde + V_tilde).

        # L = mean((x_tilde - sg(x_tilde + V_tilde))^2).
        # The target and scale are constants in backward, but prediction is not.
        loss_per_group = (
            (prediction / coordinate_scale - regression_target).square().mean(dim=(1, 2))
        )
    return loss_per_group, info


class DriftingLoss(nn.Module):
    """Structured-sample adapter; returns (reduced_loss, info).

    Input shapes: generated [B, G, *sample_shape], positive [B, P, *sample_shape].
    For action chunks use [B, G, horizon, action_dim], NOT [B, horizon, action_dim].
    Only sample_shape is flattened; B and sample counts remain separate.
    """

    def __init__(
        self,
        R_list: Sequence[float] = (0.02, 0.05, 0.2),
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.R_list = tuple(R_list)
        self.reduction = reduction

    def forward(
        self,
        generated: Tensor,
        positive: Tensor,
        negative: Tensor | None = None,
        *,
        weight_gen: Tensor | None = None,
        weight_pos: Tensor | None = None,
        weight_neg: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses, info = drift_loss(
            gen=generated.flatten(start_dim=2),
            fixed_pos=positive.flatten(start_dim=2),
            fixed_neg=None if negative is None else negative.flatten(start_dim=2),
            weight_gen=weight_gen,
            weight_pos=weight_pos,
            weight_neg=weight_neg,
            R_list=self.R_list,
        )
        if self.reduction == "mean":
            losses = losses.mean()
        elif self.reduction == "sum":
            losses = losses.sum()
        return losses, info


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt

    from mujoco_lab.learning.infrastructure.utils import build_mlp

    parser = argparse.ArgumentParser(description="Train and visualize a small 2D drifting model.")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("outputs/drifting_demo.png"))
    parser.add_argument("--show", action="store_true", help="Also open the plot window.")
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")

    torch.manual_seed(42)
    torch.set_num_threads(1)
    centers = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])

    def sample_target(count: int) -> Tensor:
        """Draw from four equally likely Gaussian clusters."""
        return centers[torch.randint(4, (count,))] + 0.25 * torch.randn(count, 2)

    class Generator(nn.Module):
        """Learn a displacement from noise, keeping initial samples spread out."""

        def __init__(self) -> None:
            super().__init__()
            self.net = build_mlp(
                input_dim=2, output_dim=2, hidden_layers=[128, 128, 128], activation_fn=nn.SiLU
            )

        def forward(self, noise: Tensor) -> Tensor:
            return noise + self.net(noise)

    generator = Generator()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    loss_module = DriftingLoss()
    # Fixed evaluation noise shows how the same latent samples move during training.
    eval_noise = torch.randn(1024, 2)
    reference = sample_target(1024)
    directions = torch.randn(2, 32)
    directions = directions / directions.norm(dim=0, keepdim=True)
    reference_projection = (reference @ directions).sort(dim=0).values
    snapshot_steps = {0, args.steps // 10, args.steps // 2, args.steps}
    snapshots = {}
    losses, distances, eval_steps = [], [], []

    for step in range(args.steps + 1):
        if step % 25 == 0 or step in snapshot_steps:
            with torch.no_grad():
                samples = generator(eval_noise)
                # Sliced W2: compare sorted 1D projections; lower means a closer match.
                projection = (samples @ directions).sort(dim=0).values
                distance = (projection - reference_projection).square().mean().sqrt().item()
            eval_steps.append(step)
            distances.append(distance)
            if step in snapshot_steps:
                snapshots[step] = samples.numpy()
                print(f"Step {step:4d}: sliced W2 = {distance:.4f}")
        if step == args.steps:
            break

        # One unconditional group: B=1, G=P=128, D=2; draw fresh training samples.
        gen = generator(torch.randn(128, 2)).unsqueeze(0)
        pos = sample_target(128).unsqueeze(0)
        loss, _ = loss_module(gen, pos)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    fig = plt.figure(figsize=(14, 7), layout="constrained")
    grid = fig.add_gridspec(2, len(snapshots))
    limit = math.ceil(
        max(reference.abs().max().item(), *(abs(s).max() for s in snapshots.values()))
    )
    for column, (step, samples) in enumerate(snapshots.items()):
        ax = fig.add_subplot(grid[0, column])
        ax.scatter(*reference.numpy().T, s=8, alpha=0.2, color="tab:blue", label="Target")
        ax.scatter(*samples.T, s=8, alpha=0.4, color="tab:orange", label="Generated")
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", title=f"Step {step}")
        ax.set_xlabel("x")
        if column == 0:
            ax.set_ylabel("y")
            ax.legend(loc="upper right", markerscale=2)

    ax = fig.add_subplot(grid[1, :2])
    ax.plot(range(1, args.steps + 1), losses, linewidth=0.8, color="tab:orange")
    ax.set(xlabel="Optimizer step", ylabel="Drifting loss", title="Normalized training objective")
    ax = fig.add_subplot(grid[1, 2:])
    ax.plot(eval_steps, distances, color="tab:blue")
    ax.set(
        xlabel="Optimizer step", ylabel="Sliced W2", title="Distribution distance (lower is better)"
    )
    fig.suptitle("Drifting: noise → MLP → four Gaussian clusters", fontsize=16)
    fig.supxlabel("The normalized drifting loss need not decrease with distribution distance.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved visualization to {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)
