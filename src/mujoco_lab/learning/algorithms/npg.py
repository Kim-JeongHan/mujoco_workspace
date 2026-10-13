"""Natural policy gradient with a rollout-only Fisher metric and KL backtracking."""

from collections.abc import Callable

import torch

from mujoco_lab.learning.policies.gaussian import GaussianPolicy


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
