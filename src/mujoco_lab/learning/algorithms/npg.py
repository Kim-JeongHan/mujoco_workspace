"""Natural policy gradient with a rollout-only Fisher metric and KL backtracking."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBatch


@dataclass
class NPGConfig:
    """Natural gradient and empirical KL step limits."""

    max_kl: float = 0.01
    damping: float = 0.1
    cg_iterations: int = 10
    backtrack_steps: int = 10

    def __post_init__(self) -> None:
        if self.max_kl <= 0 or self.damping <= 0:
            raise ValueError("max_kl and damping must be positive")
        if self.cg_iterations <= 0 or self.backtrack_steps <= 0:
            raise ValueError("CG iterations and backtrack steps must be positive")


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    iterations: int = 10,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """Approximately solve a positive-definite linear system without storing its matrix."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_norm = residual.dot(residual)
    for _ in range(iterations):
        if residual_norm <= tolerance:
            break
        product = matvec(direction)
        curvature = direction.dot(product)
        if curvature <= 0:
            break
        alpha = residual_norm / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_norm = residual.dot(residual)
        direction = residual + (next_norm / residual_norm) * direction
        residual_norm = next_norm
    return solution


def fisher_vector_product(
    policy: GaussianPolicy,
    states: torch.Tensor,
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    vector: torch.Tensor,
    *,
    damping: float,
) -> torch.Tensor:
    """Apply the Hessian of mean KL plus damping at the rollout policy parameters."""
    parameters = tuple(policy.parameters())
    kl = policy.kl_from(states, old_mean, old_std).mean()
    gradients = torch.autograd.grad(kl, parameters, create_graph=True)
    flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
    products = torch.autograd.grad(flat_gradient.dot(vector), parameters)
    return torch.cat([product.reshape(-1) for product in products]).detach() + damping * vector


def npg_step(
    policy: GaussianPolicy,
    batch: RolloutBatch,
    config: NPGConfig,
    *,
    augmentation: Callable[[], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Take one full-rollout natural gradient step, optionally adding a demo objective.

    The policy must still be the policy that collected ``batch``. Whiten rollout
    advantages once. Fisher products and KL use only rollout states, while the
    gradient and acceptance objective also include ``augmentation`` when supplied.
    Backtracking enforces the measured KL budget and improvement of that objective.
    Failed candidates leave the original policy intact.
    """
    states, actions = batch.states.detach(), batch.actions.detach()
    advantages = batch.advantages.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    old_log_probs = batch.old_log_probs.detach()
    with torch.no_grad():
        old_dist = policy(states)
        old_mean, old_std = old_dist.loc.clone(), old_dist.scale.clone()
    parameters = tuple(policy.parameters())
    original = parameters_to_vector(parameters).detach()

    def objective() -> torch.Tensor:
        ratios = (policy.log_prob(states, actions) - old_log_probs).exp()
        result = (ratios * advantages).mean()
        return result if augmentation is None else result + augmentation()

    before = objective()
    gradient = torch.cat(
        [value.reshape(-1) for value in torch.autograd.grad(before, parameters)]
    ).detach()
    if not torch.isfinite(gradient).all():
        raise RuntimeError("nonfinite policy gradient")

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return fisher_vector_product(
            policy, states, old_mean, old_std, vector, damping=config.damping
        )

    direction = conjugate_gradient(matvec, gradient, iterations=config.cg_iterations)
    curvature = gradient.dot(direction)
    accepted = False
    after = before.detach().item()
    measured_kl = 0.0
    fraction = 0.0
    if curvature > 0:
        full_step = direction * torch.sqrt(2 * config.max_kl / curvature)
        try:
            with torch.no_grad():
                for attempt in range(config.backtrack_steps):
                    fraction = 0.5**attempt
                    vector_to_parameters(original + fraction * full_step, parameters)
                    candidate = objective()
                    kl = policy.kl_from(states, old_mean, old_std).mean()
                    if (
                        torch.isfinite(candidate)
                        and torch.isfinite(kl)
                        and kl <= config.max_kl
                        and candidate > before.detach()
                    ):
                        accepted = True
                        after, measured_kl = candidate.item(), kl.item()
                        break
        finally:
            if not accepted:
                with torch.no_grad():
                    vector_to_parameters(original, parameters)
                fraction = 0.0
    return {
        "policy/objective_before": before.detach().item(),
        "policy/objective_after": after,
        "policy/kl": measured_kl,
        "policy/gradient_norm": gradient.norm().item(),
        "policy/accepted": float(accepted),
        "policy/step_fraction": fraction,
    }
