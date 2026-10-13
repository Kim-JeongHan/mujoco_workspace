"""Initialize Gaussian policy weights from a behavior-cloning checkpoint."""

import math
from pathlib import Path
from typing import Any

import torch

from mujoco_lab.learning.checkpoint import load_checkpoint
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy


def initialize_actor(
    init_from: Path,
    *,
    initial_log_std: float = -2.0,
    device: str = "cpu",
) -> tuple[GaussianPolicy, Normalizer, dict[str, Any]]:
    """Copy BC weights into the Gaussian mean and retain BC normalization.

    The actor preserves BC observation and chunk dimensions. Returned metadata
    describes the source BC checkpoint, not a resumable agent checkpoint.
    ``initial_log_std`` sets exploration in normalized action coordinates.
    """
    if not math.isfinite(initial_log_std):
        raise ValueError("initial_log_std must be finite")
    bc_policy, normalizer, metadata = load_checkpoint(init_from)
    if not isinstance(bc_policy, MSEPolicy):
        raise ValueError("Agent initialization requires an MSE BC checkpoint")

    actor = GaussianPolicy(
        state_dim=bc_policy.state_dim,
        action_dim=bc_policy.action_dim,
        chunk_size=bc_policy.chunk_size,
        hidden_dims=tuple(metadata["architecture"]["hidden_dims"]),
    )
    actor.net.load_state_dict(bc_policy.net[:-1].state_dict())
    actor.mean_head.load_state_dict(bc_policy.net[-1].state_dict())
    with torch.no_grad():
        actor.log_std.fill_(initial_log_std)
    actor.to(device)
    actor.train()

    return actor, normalizer, metadata
