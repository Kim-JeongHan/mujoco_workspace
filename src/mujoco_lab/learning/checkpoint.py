"""BC inference and resumable RL checkpoints with shared serialization helpers."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from mujoco_lab.learning.agents import DAPGAgent, DDPGAgent
from mujoco_lab.learning.config import (
    BCPolicyTrainConfig,
    CriticConfig,
    DAPGConfig,
    DDPGConfig,
    DiffusionBCTrainConfig,
    DiffusionSchedulerConfig,
    EvaluationConfig,
    FlowBCTrainConfig,
    HERConfig,
    MSEBCTrainConfig,
    NPGConfig,
)
from mujoco_lab.learning.datasets.normalizer import Normalizer, RunningMoments
from mujoco_lab.learning.metadata import (
    CheckpointMetadata,
    DatasetMetadata,
    EnvironmentMetadata,
    ObservationMetadata,
    PolicyMetadata,
)
from mujoco_lab.learning.models.critic import QCritic, ValueCritic
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.diffusion import DiffusionPolicy
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.policies.flow import FlowMatchingPolicy
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.rollout.replay_buffer import HERReplayBuffer


def save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    """Write tensors and primitive metadata without replacing an existing file."""
    with Path(path).open("xb") as file:
        torch.save(payload, file)


def load_payload(path: str | Path) -> dict[str, Any]:
    """Read tensor-only checkpoint data on CPU before rebuilding training objects."""
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(
    path: str | Path,
    model: BasePolicy,
    normalizer: Normalizer,
    config: BCPolicyTrainConfig,
    *,
    optimizer_step: int,
    dataset: DatasetMetadata | None = None,
    environment: EnvironmentMetadata,
    observation: ObservationMetadata | None = None,
) -> CheckpointMetadata:
    """Save a reloadable inference policy without pickling Python model objects."""
    metadata = checkpoint_metadata(
        model,
        normalizer,
        config,
        optimizer_step=optimizer_step,
        dataset=dataset,
        environment=environment,
        observation=observation,
    )
    payload = {
        **metadata.to_dict(),
        "model_state": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "normalizer": normalizer.state_dict(),
    }
    save_payload(path, payload)
    return metadata


def checkpoint_metadata(
    model: BasePolicy,
    normalizer: Normalizer,
    config: BCPolicyTrainConfig,
    *,
    optimizer_step: int,
    dataset: DatasetMetadata | None = None,
    environment: EnvironmentMetadata,
    observation: ObservationMetadata | None = None,
) -> CheckpointMetadata:
    """Describe the live policy with the same contract as its saved checkpoint."""
    observation = observation or ObservationMetadata(len(normalizer.state_mean))
    policy = PolicyMetadata(
        policy_type=config.policy_type,
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        chunk_size=model.chunk_size,
        obs_horizon=config.obs_horizon,
        execution_horizon=config.execution_horizon,
        hidden_dims=(
            config.hidden_dims if isinstance(config, (MSEBCTrainConfig, FlowBCTrainConfig)) else ()
        ),
        sampling_num_steps=(
            config.sampling_num_steps
            if isinstance(config, (FlowBCTrainConfig, DiffusionBCTrainConfig))
            else 1
        ),
        flow_time_embed_dim=model.time_embed_dim if isinstance(model, FlowMatchingPolicy) else 256,
        diffusion_down_dims=tuple(model.denoiser.down_dims)
        if isinstance(model, DiffusionPolicy)
        else (),
        diffusion_time_embed_dim=model.denoiser.time_embed_dim
        if isinstance(model, DiffusionPolicy)
        else 0,
        diffusion_scheduler_config=(
            {key: value for key, value in model.scheduler.config.items() if not key.startswith("_")}
            if isinstance(model, DiffusionPolicy)
            else {}
        ),
    )
    return CheckpointMetadata(
        policy,
        observation,
        environment,
        dataset,
        train_config=config_dict(config),
        optimizer_step=optimizer_step,
    )


def load_checkpoint(path: str | Path) -> tuple[BasePolicy, Normalizer, CheckpointMetadata]:
    """Restore policy and normalization on CPU with ``weights_only=True``."""
    payload = load_payload(path)
    metadata = CheckpointMetadata.from_dict(payload)
    architecture = metadata.policy
    policy_config: BCPolicyTrainConfig
    if architecture.policy_type == "mse":
        policy_config = MSEBCTrainConfig(
            policy_type="mse",
            chunk_size=architecture.chunk_size,
            hidden_dims=tuple(architecture.hidden_dims),
        )
    elif architecture.policy_type == "flow":
        policy_config = FlowBCTrainConfig(
            policy_type="flow",
            chunk_size=architecture.chunk_size,
            hidden_dims=tuple(architecture.hidden_dims),
            time_embed_dim=architecture.flow_time_embed_dim,
        )
    elif architecture.policy_type == "diffusion":
        scheduler_settings = dict(architecture.diffusion_scheduler_config)
        scheduler_settings.pop("prediction_type")
        policy_config = DiffusionBCTrainConfig(
            policy_type="diffusion",
            chunk_size=architecture.chunk_size,
            down_dims=tuple(architecture.diffusion_down_dims),
            time_embed_dim=architecture.diffusion_time_embed_dim,
            scheduler=DiffusionSchedulerConfig(**scheduler_settings),
        )
    else:
        raise ValueError(f"Unsupported BC policy: {architecture.policy_type}")
    model = build_policy(
        policy_config,
        state_dim=architecture.state_dim,
        action_dim=architecture.action_dim,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    normalizer = Normalizer.from_state_dict(payload["normalizer"])
    return model, normalizer, metadata


@dataclass
class TrainingState[AgentType: (DAPGAgent, DDPGAgent)](ABC):
    """Shared learned state and serialization at a seeded rollout boundary."""

    agent: AgentType
    normalizer: Normalizer
    metadata: CheckpointMetadata
    env_steps: int = 0
    rng: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        """Snapshot learned state and capture the current training RNG."""
        return {
            **self.agent.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "metadata": self.metadata.to_dict(),
            "config": config_dict(self.agent.config),
            "env_steps": self.env_steps,
            "rng": capture_rng(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any], *, device: str = "cpu") -> Self:
        """Rebuild learned state while deferring RNG restoration until setup finishes."""
        metadata = CheckpointMetadata.from_dict(payload["metadata"])
        agent = cls.create_agent(payload, metadata, device=device)
        agent.load_state_dict(payload)
        state = cls(
            agent=agent,
            normalizer=Normalizer.from_state_dict(
                payload["normalizer"], state_clip=payload.get("state_clip")
            ),
            metadata=metadata,
            env_steps=payload["env_steps"],
            rng=payload["rng"],
        )
        state.restore_extra_state(payload)
        return state

    @classmethod
    @abstractmethod
    def create_agent(
        cls, payload: dict[str, Any], metadata: CheckpointMetadata, *, device: str
    ) -> AgentType:
        """Construct the algorithm's networks and optimizer objects before loading weights."""

    def restore_extra_state(self, payload: dict[str, Any]) -> None:
        """Restore any algorithm-specific state beyond the shared learning state."""
        return None


class DAPGTrainingState(TrainingState[DAPGAgent]):
    """DAPG learning state with a Gaussian policy and state-value baseline."""

    @classmethod
    def create_agent(
        cls, payload: dict[str, Any], metadata: CheckpointMetadata, *, device: str
    ) -> DAPGAgent:
        settings = dict(payload["config"])
        for key in ("init_from", "resume", "data_dir", "output_dir"):
            if settings[key] is not None:
                settings[key] = Path(settings[key])
        settings["device"] = device
        settings["npg"] = NPGConfig(**settings["npg"])
        critic_settings = dict(settings["critic"])
        critic_settings["hidden_dims"] = tuple(critic_settings["hidden_dims"])
        settings["critic"] = CriticConfig(**critic_settings)
        config = DAPGConfig(**settings)
        architecture = metadata.policy
        actor = GaussianPolicy(
            architecture.state_dim,
            architecture.action_dim,
            architecture.chunk_size,
            hidden_dims=tuple(architecture.hidden_dims),
        ).to(device)
        critic = ValueCritic(
            actor.state_dim,
            config.critic.hidden_dims,
            config.critic.learning_rate,
            device=device,
        )
        return DAPGAgent(actor, critic, config)


@dataclass
class DDPGTrainingState(TrainingState[DDPGAgent]):
    """DDPG learning state extended with online statistics and episode replay."""

    moments: RunningMoments = field(init=False)
    replay: HERReplayBuffer = field(init=False)
    episodes: int = 0

    def __post_init__(self) -> None:
        self.moments = RunningMoments(
            len(self.normalizer.state_mean),
            self.metadata.observation.rotation_indices,
        )
        self.replay = HERReplayBuffer(self.agent.config.replay_capacity)

    def state_dict(self) -> dict[str, Any]:
        """Extend shared learning state with clipping, statistics, replay, and episodes."""
        return {
            **super().state_dict(),
            "state_clip": self.normalizer.state_clip,
            "moments": {
                "count": self.moments.count,
                "mean": torch.from_numpy(self.moments.mean.copy()),
                "m2": torch.from_numpy(self.moments.m2.copy()),
            },
            "replay": self.replay.state_dict(),
            "episodes": self.episodes,
        }

    @classmethod
    def create_agent(
        cls, payload: dict[str, Any], metadata: CheckpointMetadata, *, device: str
    ) -> DDPGAgent:
        settings = dict(payload["config"])
        for key in ("resume", "output_dir"):
            if settings[key] is not None:
                settings[key] = Path(settings[key])
        settings["device"] = device
        for key in ("hidden_dims", "critic_hidden_dims"):
            settings[key] = tuple(settings[key])
        settings["her"] = HERConfig(**settings["her"])
        settings["evaluation"] = EvaluationConfig(**settings["evaluation"])
        config = DDPGConfig(**settings)
        architecture = metadata.policy
        policy = MSEPolicy(
            architecture.state_dim,
            architecture.action_dim,
            architecture.chunk_size,
            architecture.hidden_dims,
            bounded=architecture.bounded_actions,
        ).to(device)
        critic = QCritic(policy.state_dim, policy.action_dim, config.critic_hidden_dims).to(device)
        return DDPGAgent(policy, critic, config)

    def restore_extra_state(self, payload: dict[str, Any]) -> None:
        self.moments.count = payload["moments"]["count"]
        self.moments.mean[:] = payload["moments"]["mean"].numpy()
        self.moments.m2[:] = payload["moments"]["m2"].numpy()
        self.replay.load_state_dict(payload["replay"])
        self.episodes = payload["episodes"]


def config_dict(config: BCPolicyTrainConfig | DAPGConfig | DDPGConfig) -> dict[str, Any]:
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


def save_training_checkpoint[AgentType: (DAPGAgent, DDPGAgent)](
    path: Path, state: TrainingState[AgentType]
) -> None:
    """Save a training state's serialized data without replacing an existing file."""
    save_payload(path, state.state_dict())


def load_training_checkpoint(path: Path, *, device: str = "cpu") -> DAPGTrainingState:
    """Read a DAPG checkpoint and rebuild its learned state."""
    return DAPGTrainingState.from_state_dict(load_payload(path), device=device)


def restore_rng(rng: dict[str, Any]) -> None:
    """Restore sampling and minibatch RNG after rebuilding training objects."""
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    random.setstate(rng["python"])
    name, keys, position, has_gauss, cached_gauss = rng["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss))


def save_ddpg_checkpoint(path: Path, state: DDPGTrainingState) -> None:
    """Save DDPG learning state through the shared training checkpoint writer."""
    save_training_checkpoint(path, state)


def load_ddpg_checkpoint(path: Path, *, device: str = "cpu") -> DDPGTrainingState:
    """Read a DDPG checkpoint and rebuild its learned state and episode replay."""
    return DDPGTrainingState.from_state_dict(load_payload(path), device=device)
