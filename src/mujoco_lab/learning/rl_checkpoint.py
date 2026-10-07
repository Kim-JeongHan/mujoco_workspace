"""Iteration-boundary DAPG checkpoints with critic, optimizer, and random state."""

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_lab.learning.algorithms.npg import NPGConfig
from mujoco_lab.learning.config.dapg import DAPGConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy


@dataclass
class TrainingState:
    """All learned state needed to continue at the next seeded rollout boundary."""

    actor: GaussianPolicy
    critic: ValueCritic
    normalizer: Normalizer
    metadata: dict[str, Any]
    config: DAPGConfig
    iteration: int = 0
    env_steps: int = 0
    rng: dict[str, Any] | None = None


def config_dict(config: DAPGConfig) -> dict[str, Any]:
    """Serialize configuration using only primitives accepted by JSON and weights_only."""
    result = asdict(config)
    for key in ("init_from", "resume", "data_dir", "output_dir"):
        if result[key] is not None:
            result[key] = str(result[key])
    return result


def save_training_checkpoint(path: Path, state: TrainingState) -> None:
    """Save exclusively; NPG has no persistent actor optimizer state."""
    np_rng = np.random.get_state()
    payload = {
        "format": "mujoco_lab.dapg.v1",
        "actor": state.actor.state_dict(),
        "critic": state.critic.state_dict(),
        "critic_optimizer": state.critic.optimizer.state_dict(),
        "normalizer": {
            name: torch.from_numpy(getattr(state.normalizer, name).copy())
            for name in ("state_mean", "state_std", "action_mean", "action_std")
        },
        "metadata": state.metadata,
        "config": config_dict(state.config),
        "iteration": state.iteration,
        "env_steps": state.env_steps,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (np_rng[0], np_rng[1].tolist(), *np_rng[2:]),
        },
    }
    with path.open("xb") as file:
        torch.save(payload, file)


def load_training_checkpoint(path: Path, *, device: str = "cpu") -> TrainingState:
    """Reconstruct networks and optimizer; defer RNG restoration until setup is finished."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != "mujoco_lab.dapg.v1":
        raise ValueError("unsupported DAPG checkpoint format")
    settings = payload["config"]
    for key in ("init_from", "resume", "data_dir", "output_dir"):
        if settings[key] is not None:
            settings[key] = Path(settings[key])
    settings["device"] = device
    settings["npg"] = NPGConfig(**settings["npg"])
    config = DAPGConfig(**settings)
    architecture = payload["metadata"]["architecture"]
    actor = GaussianPolicy(
        architecture["state_dim"],
        architecture["action_dim"],
        architecture["chunk_size"],
        hidden_dims=tuple(architecture["hidden_dims"]),
    ).to(device)
    actor.load_state_dict(payload["actor"])
    critic = ValueCritic(
        actor.state_dim,
        config.critic_n_layers,
        config.critic_layer_size,
        config.critic_learning_rate,
        device=device,
    )
    critic.load_state_dict(payload["critic"])
    critic.optimizer.load_state_dict(payload["critic_optimizer"])
    normalizer = Normalizer(
        **{name: tensor.numpy() for name, tensor in payload["normalizer"].items()}
    )
    return TrainingState(
        actor,
        critic,
        normalizer,
        payload["metadata"],
        config,
        payload["iteration"],
        payload["env_steps"],
        payload["rng"],
    )


def restore_rng(rng: dict[str, Any]) -> None:
    """Restore sampling and minibatch RNG after rebuilding training objects."""
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    random.setstate(rng["python"])
    name, keys, position, has_gauss, cached_gauss = rng["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
