"""Inference-only BC checkpoints containing tensors and primitive metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.replay import replay_action_repeat
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.policies.flow import FlowMatchingPolicy


def save_checkpoint(
    path: str | Path,
    model: BasePolicy,
    normalizer: Normalizer,
    config: TrainConfig,
    *,
    optimizer_step: int,
    dataset_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a reloadable inference policy without pickling Python model objects."""
    metadata = checkpoint_metadata(
        model, normalizer, config, optimizer_step=optimizer_step, dataset_metadata=dataset_metadata
    )
    payload = {
        "format_version": 1,
        **metadata,
        "model_state": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "normalizer": {
            name: torch.from_numpy(getattr(normalizer, name).copy())
            for name in ("state_mean", "state_std", "action_mean", "action_std")
        },
    }
    with Path(path).open("xb") as file:
        torch.save(payload, file)


def checkpoint_metadata(
    model: BasePolicy,
    normalizer: Normalizer,
    config: TrainConfig,
    *,
    optimizer_step: int,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the live policy with the same contract as its saved checkpoint."""
    frame_dim = int(normalizer.state_mean.shape[0])
    if model.state_dim != frame_dim * config.obs_horizon:
        raise ValueError("Policy input does not match normalizer frame size and obs_horizon")
    if model.action_dim != normalizer.action_mean.shape[0]:
        raise ValueError("Policy action dimension does not match normalizer")
    replay_repeat = replay_action_repeat((dataset_metadata or {}).get("replay") or {})
    if replay_repeat != config.physics_steps_per_action:
        raise ValueError(
            "Dataset physics_steps_per_action differs from training config; "
            "resample or recollect with --physics-steps-per-action matching the config"
        )
    architecture = {
        "policy_type": config.policy_type,
        "frame_dim": frame_dim,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "chunk_size": model.chunk_size,
        "hidden_dims": list(config.hidden_dims),
        "obs_horizon": config.obs_horizon,
        "execution_horizon": config.execution_horizon,
        "physics_steps_per_action": config.physics_steps_per_action,
    }
    if isinstance(model, FlowMatchingPolicy):
        architecture["flow_time_embed_dim"] = model.time_embed_dim
    settings = asdict(config)
    settings["data_dir"] = str(config.data_dir)
    settings["output_dir"] = str(config.output_dir)
    return {
        "architecture": architecture,
        "train_config": settings,
        "dataset_metadata": json.loads(json.dumps(dataset_metadata or {}, allow_nan=False)),
        "optimizer_step": optimizer_step,
    }


def checkpoint_action_repeat(metadata: dict[str, Any]) -> int:
    """Resolve action cadence across checkpoint, training config, and replay data."""
    architecture = metadata.get("architecture") or {}
    training = metadata.get("train_config") or {}
    replay = (metadata.get("dataset_metadata") or {}).get("replay") or {}
    replay_repeat = replay_action_repeat(replay)
    architecture_repeat = architecture.get("physics_steps_per_action", 1) if architecture else None
    training_repeat = training.get("physics_steps_per_action", 1) if training else None
    for source, repeat in (
        ("architecture", architecture_repeat),
        ("train_config", training_repeat),
    ):
        if repeat is None:
            continue
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat <= 0:
            raise ValueError(f"Checkpoint {source} physics_steps_per_action must be positive")
    if (architecture_repeat is not None and architecture_repeat != replay_repeat) or (
        training_repeat is not None and training_repeat != replay_repeat
    ):
        raise ValueError(
            "Checkpoint architecture, train_config, and dataset replay "
            "physics_steps_per_action differ; resample or recollect with a matching "
            "--physics-steps-per-action"
        )
    return replay_repeat


def load_checkpoint(path: str | Path) -> tuple[BasePolicy, Normalizer, dict[str, Any]]:
    """Restore policy and normalization on CPU with ``weights_only=True``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    architecture = dict(payload["architecture"])
    architecture.setdefault("physics_steps_per_action", 1)
    model = build_policy(
        architecture["policy_type"],
        state_dim=architecture["state_dim"],
        action_dim=architecture["action_dim"],
        chunk_size=architecture["chunk_size"],
        hidden_dims=tuple(architecture["hidden_dims"]),
        flow_time_embed_dim=architecture.get("flow_time_embed_dim"),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    stats = payload["normalizer"]
    normalizer = Normalizer(
        state_mean=stats["state_mean"].numpy().copy(),
        state_std=stats["state_std"].numpy().copy(),
        action_mean=stats["action_mean"].numpy().copy(),
        action_std=stats["action_std"].numpy().copy(),
    )
    if (
        normalizer.state_mean.shape != (architecture["frame_dim"],)
        or normalizer.action_mean.shape != (architecture["action_dim"],)
        or architecture["state_dim"] != architecture["frame_dim"] * architecture["obs_horizon"]
    ):
        raise ValueError("Checkpoint dimensions are inconsistent")
    metadata = {
        "architecture": architecture,
        "train_config": dict(payload["train_config"]),
        "dataset_metadata": payload.get("dataset_metadata", {}),
        "optimizer_step": payload["optimizer_step"],
    }
    metadata["train_config"].setdefault("physics_steps_per_action", 1)
    checkpoint_action_repeat(metadata)
    return model, normalizer, metadata
