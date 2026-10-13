"""Natural-gradient agent with rollout-only Fisher products and KL backtracking."""

from collections.abc import Callable

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mujoco_lab.learning.agents.base import OnPolicyAgent
from mujoco_lab.learning.algorithms import npg
from mujoco_lab.learning.config.critic import CriticConfig
from mujoco_lab.learning.config.npg import NPGConfig
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBatch


class NPGAgent(OnPolicyAgent):
    """Own the policy, critic, and natural-gradient solver configuration."""

    def __init__(
        self,
        policy: GaussianPolicy,
        critic: ValueCritic,
        npg_config: NPGConfig,
        *,
        critic_config: CriticConfig | None = None,
    ) -> None:
        super().__init__(policy, critic, CriticConfig() if critic_config is None else critic_config)
        self.npg_config = npg_config

    def update_policy(
        self,
        batch: RolloutBatch,
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
        policy, config = self.policy, self.npg_config
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
            return npg.fisher_vector_product(
                policy, states, old_mean, old_std, vector, damping=config.damping
            )

        direction = npg.conjugate_gradient(matvec, gradient, iterations=config.cg_iterations)
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
