"""Iteration-boundary DAPG checkpoints with critic, optimizer, and random state."""

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_lab.learning.agents import DAPGAgent, DDPGAgent
from mujoco_lab.learning.config.critic import CriticConfig
from mujoco_lab.learning.config.dapg import DAPGConfig
from mujoco_lab.learning.config.ddpg import DDPGConfig, HERConfig
from mujoco_lab.learning.config.evaluation import EvaluationConfig
from mujoco_lab.learning.config.npg import NPGConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer, RunningMoments
from mujoco_lab.learning.models.critic import QCritic, ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.rollout.replay_buffer import HERReplayBuffer


@dataclass
class TrainingState:
    """All learned state needed to continue at the next seeded rollout boundary."""

    agent: DAPGAgent
    normalizer: Normalizer
    metadata: dict[str, Any]
    env_steps: int = 0
    rng: dict[str, Any] | None = None


def config_dict(config: DAPGConfig | DDPGConfig) -> dict[str, Any]:
    """Serialize configuration using only primitives accepted by JSON and weights_only."""
    result = asdict(config)
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def capture_rng() -> dict[str, Any]:
    """Capture training RNG independently of model reconstruction and evaluation."""
    np_rng = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (np_rng[0], np_rng[1].tolist(), *np_rng[2:]),
    }


def save_training_checkpoint(path: Path, state: TrainingState) -> None:
    """Save exclusively; NPG has no persistent actor optimizer state."""
    payload = {
        "format": "mujoco_lab.dapg.v1",
        **state.agent.state_dict(),
        "normalizer": {
            name: torch.from_numpy(getattr(state.normalizer, name).copy())
            for name in ("state_mean", "state_std", "action_mean", "action_std")
        },
        "metadata": state.metadata,
        "config": config_dict(state.agent.config),
        "env_steps": state.env_steps,
        "rng": capture_rng(),
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
    settings["critic"]["hidden_dims"] = tuple(settings["critic"]["hidden_dims"])
    settings["critic"] = CriticConfig(**settings["critic"])
    config = DAPGConfig(**settings)
    architecture = payload["metadata"]["architecture"]
    actor = GaussianPolicy(
        architecture["state_dim"],
        architecture["action_dim"],
        architecture["chunk_size"],
        hidden_dims=tuple(architecture["hidden_dims"]),
    ).to(device)
    critic = ValueCritic(
        actor.state_dim,
        config.critic.hidden_dims,
        config.critic.learning_rate,
        device=device,
    )
    agent = DAPGAgent(actor, critic, config)
    agent.load_state_dict(payload)
    normalizer = Normalizer(
        **{name: tensor.numpy() for name, tensor in payload["normalizer"].items()}
    )
    return TrainingState(
        agent,
        normalizer,
        payload["metadata"],
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


@dataclass
class DDPGTrainingState:
    """Episode-boundary learning state including normalization and replay contents."""

    agent: DDPGAgent
    normalizer: Normalizer
    moments: RunningMoments
    replay: HERReplayBuffer
    metadata: dict[str, Any]
    env_steps: int = 0
    episodes: int = 0
    rng: dict[str, Any] | None = None


def save_ddpg_checkpoint(path: Path, state: DDPGTrainingState) -> None:
    """Save target networks, both optimizers, replay, statistics, and RNG exclusively."""
    payload = {
        "format": "mujoco_lab.ddpg.v1",
        **state.agent.state_dict(),
        "normalizer": {
            name: torch.from_numpy(getattr(state.normalizer, name).copy())
            for name in ("state_mean", "state_std", "action_mean", "action_std")
        },
        "state_clip": state.normalizer.state_clip,
        "moments": {
            "count": state.moments.count,
            "mean": torch.from_numpy(state.moments.mean.copy()),
            "m2": torch.from_numpy(state.moments.m2.copy()),
        },
        "replay": state.replay.state_dict(),
        "metadata": state.metadata,
        "config": config_dict(state.agent.config),
        "env_steps": state.env_steps,
        "episodes": state.episodes,
        "rng": capture_rng(),
    }
    with path.open("xb") as file:
        torch.save(payload, file)


def load_ddpg_checkpoint(path: Path, *, device: str = "cpu") -> DDPGTrainingState:
    """Restore a training boundary; restore saved RNG after environment construction."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != "mujoco_lab.ddpg.v1":
        raise ValueError("unsupported DDPG checkpoint format")
    settings = payload["config"]
    for key in ("resume", "output_dir"):
        if settings[key] is not None:
            settings[key] = Path(settings[key])
    settings["device"] = device
    for key in ("hidden_dims", "critic_hidden_dims"):
        settings[key] = tuple(settings[key])
    settings["her"] = HERConfig(**settings["her"])
    settings["evaluation"] = EvaluationConfig(**settings["evaluation"])
    config = DDPGConfig(**settings)
    architecture = payload["metadata"]["architecture"]
    policy = MSEPolicy(
        architecture["state_dim"], architecture["action_dim"], 1, config.hidden_dims, bounded=True
    ).to(device)
    critic = QCritic(policy.state_dim, policy.action_dim, config.critic_hidden_dims).to(device)
    agent = DDPGAgent(policy, critic, config)
    agent.load_state_dict(payload)
    normalizer = Normalizer(
        **{name: tensor.numpy() for name, tensor in payload["normalizer"].items()},
        state_clip=payload["state_clip"],
    )
    moments = RunningMoments(
        len(normalizer.state_mean),
        payload["metadata"]["dataset_metadata"]["observation"]["rotation_indices"],
    )
    moments.count = payload["moments"]["count"]
    moments.mean[:] = payload["moments"]["mean"].numpy()
    moments.m2[:] = payload["moments"]["m2"].numpy()
    replay = HERReplayBuffer(config.replay_capacity)
    replay.load_state_dict(payload["replay"])
    return DDPGTrainingState(
        agent,
        normalizer,
        moments,
        replay,
        payload["metadata"],
        payload["env_steps"],
        payload["episodes"],
        payload["rng"],
    )
