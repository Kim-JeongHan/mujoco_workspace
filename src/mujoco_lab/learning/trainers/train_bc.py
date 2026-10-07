"""Offline behavior cloning from already collected demonstration episodes."""

from collections.abc import Callable, Sequence
from time import perf_counter

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.sequence import ChunkDataset
from mujoco_lab.learning.infrastructure.utils import set_seed
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.factory import build_policy


def run_training(
    config: TrainConfig,
    train_episodes: Sequence[Episode],
    validation_episodes: Sequence[Episode] = (),
    *,
    logger: Logger | None = None,
    evaluate: Callable[[BasePolicy, Normalizer, int], None] | None = None,
) -> tuple[BasePolicy, Normalizer]:
    """Train BC from episodes and return the policy and normalizer.

    Episodes should be validated with ``datasets.load_episodes`` and split before
    training. Feature dimensions and observation layouts are checked here without
    rescanning observation and action arrays. Inference checkpoints are saved separately;
    State statistics are fitted on raw training frames and applied to each frame before
    history flattening. An optional callback evaluates the current in-memory model.
    Flow validation loss is a sampled flow-matching objective, not task success.
    Callers supplying episodes directly must validate their arrays, observation metadata,
    and cadence first.
    """
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if not 1 <= config.execution_horizon <= config.chunk_size:
        raise ValueError("execution_horizon must be between 1 and chunk_size")
    if config.ema_decay is not None and not 0 <= config.ema_decay < 1:
        raise ValueError("ema_decay must be None or in [0, 1)")
    if not train_episodes:
        raise ValueError("Training requires at least one episode")
    set_seed(config.seed)
    frame_dim = train_episodes[0].states.shape[1]
    action_dim = train_episodes[0].actions.shape[1]
    rotation_indices = train_episodes[0].rotation_indices
    rotation_layout = set(rotation_indices)
    for episode in (*train_episodes, *validation_episodes):
        if (episode.states.shape[1], episode.actions.shape[1]) != (frame_dim, action_dim):
            raise ValueError("Training and validation episode feature dimensions must match")
        if set(episode.rotation_indices) != rotation_layout:
            raise ValueError("Training and validation observation layouts must match")

    train_states = np.concatenate([episode.states[:-1] for episode in train_episodes])
    train_actions = np.concatenate([episode.actions for episode in train_episodes])
    normalizer = Normalizer.from_data(
        train_states,
        train_actions,
        state_passthrough_indices=rotation_indices,
    )
    del train_states, train_actions
    train_dataset = ChunkDataset(
        train_episodes, config.chunk_size, normalizer, obs_horizon=config.obs_horizon
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    validation_dataset = ChunkDataset(
        validation_episodes, config.chunk_size, normalizer, obs_horizon=config.obs_horizon
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_policy(
        config.policy_type,
        state_dim=frame_dim * config.obs_horizon,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
        flow_time_embed_dim=config.flow_time_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    averaged = None
    if config.ema_decay is not None:
        averaged = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay))
        averaged.module.requires_grad_(False)
        averaged.module.eval()
    # AveragedModel copies our BasePolicy but exposes its module as the base Module type.
    evaluation_model: BasePolicy = (  # ty: ignore[invalid-assignment]
        model if averaged is None else averaged.module
    )

    @torch.compile(options={"fallback_random": True})
    def train_step(state, action_chunk):
        loss = model.compute_loss(state, action_chunk)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    started = perf_counter()
    step = 0
    last_evaluation_step = 0
    logged_loss = 0.0
    logged_samples = 0
    for epoch in range(config.num_epochs):
        model.train()
        for state, action_chunk in train_loader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            loss = train_step(state, action_chunk)
            if averaged is not None:
                averaged.update_parameters(model)

            step += 1
            logged_loss += loss.item() * state.shape[0]
            logged_samples += state.shape[0]
            if step % config.log_interval == 0:
                value = logged_loss / logged_samples
                if logger is not None:
                    logger.log(
                        {
                            "train/loss": value,
                            "epoch": epoch + 1,
                            "lr": optimizer.param_groups[0]["lr"],
                            "elapsed_seconds": perf_counter() - started,
                        },
                        step=step,
                    )
                logged_loss = 0.0
                logged_samples = 0
            if (
                evaluate is not None
                and config.eval_interval > 0
                and step % config.eval_interval == 0
            ):
                evaluate(evaluation_model, normalizer, step)
                last_evaluation_step = step

        was_training = evaluation_model.training
        evaluation_model.eval()
        validation_loss = 0.0
        validation_samples = 0
        with torch.no_grad():
            for state, action_chunk in validation_loader:
                state = state.to(device)
                action_chunk = action_chunk.to(device)
                loss = evaluation_model.compute_loss(state, action_chunk)
                validation_loss += loss.item() * state.shape[0]
                validation_samples += state.shape[0]
        evaluation_model.train(was_training)
        if validation_samples:
            value = validation_loss / validation_samples
            if logger is not None:
                logger.log(
                    {
                        "validation/loss": value,
                        "epoch": epoch + 1,
                        "elapsed_seconds": perf_counter() - started,
                    },
                    step=step,
                )

    if logged_samples:
        value = logged_loss / logged_samples
        if logger is not None:
            logger.log(
                {
                    "train/loss": value,
                    "epoch": config.num_epochs,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": perf_counter() - started,
                },
                step=step,
            )
    if evaluate is not None and config.eval_interval > 0 and step != last_evaluation_step:
        evaluate(evaluation_model, normalizer, step)
    return evaluation_model, normalizer
