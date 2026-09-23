"""Offline behavior cloning from already collected demonstration episodes."""

from collections.abc import Callable, Sequence
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.loading import validate_episode_cadence
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.sequence import ChunkDataset
from mujoco_lab.learning.infrastructure.utils import set_seed
from mujoco_lab.learning.logging import Logger
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.utils.logger import Logger as ConsoleLogger


def _cube_stack_rotation_indices(episodes: Sequence[Episode], frame_dim: int) -> list[int]:
    """Locate rot6d features in single-robot cube-stack observations."""
    replays = [episode.metadata.get("replay") or {} for episode in episodes]
    cube_replays = [replay for replay in replays if replay.get("scene") == "cube_stack"]
    if not cube_replays:
        return []
    if len(cube_replays) != len(episodes):
        raise ValueError("Training episodes mix cube-stack and other observation layouts")
    cubes = cube_replays[0].get("cubes")
    if not isinstance(cubes, int) or isinstance(cubes, bool) or cubes <= 0:
        raise ValueError("Cube-stack replay metadata must specify a positive cube count")
    if any(replay.get("cubes") != cubes for replay in cube_replays):
        raise ValueError("Training episodes have different cube counts")
    if frame_dim != 24 + 15 * cubes:
        raise ValueError("Cube-stack observation dimension does not match replay cube count")
    return [
        *range(18, 24),
        *(index for cube in range(cubes) for index in range(27 + 15 * cube, 33 + 15 * cube)),
    ]


def run_training(
    config: TrainConfig,
    train_episodes: Sequence[Episode],
    validation_episodes: Sequence[Episode] = (),
    *,
    logger: Logger | None = None,
    evaluate: Callable[[BasePolicy, Normalizer, int], None] | None = None,
) -> tuple[BasePolicy, Normalizer]:
    """Train BC from pre-split episodes and return the policy and normalizer.

    Episodes can be loaded and split with ``datasets.load_episodes`` and
    ``datasets.split_episodes``. Inference checkpoints are saved separately;
    State statistics are fitted on raw training frames and applied to each frame before
    history flattening. An optional callback evaluates the current in-memory model.
    Flow validation loss is a sampled flow-matching objective, not task success.
    """
    config.validate()
    if not train_episodes:
        raise ValueError("Training requires at least one episode")
    validate_episode_cadence(
        [*train_episodes, *validation_episodes], config.physics_steps_per_action
    )
    set_seed(config.seed)
    frame_dim = train_episodes[0].states.shape[1]
    action_dim = train_episodes[0].actions.shape[1]
    rotation_indices = _cube_stack_rotation_indices(
        [*train_episodes, *validation_episodes], frame_dim
    )

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
    if len(train_dataset) == 0:
        raise ValueError("Training episodes are shorter than chunk_size")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    validation_loader = None
    if validation_episodes:
        validation_dataset = ChunkDataset(
            validation_episodes, config.chunk_size, normalizer, obs_horizon=config.obs_horizon
        )
        if len(validation_dataset) == 0:
            raise ValueError("Validation episodes are shorter than chunk_size")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    console = ConsoleLogger()
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
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(state, action_chunk)
            loss.backward()
            optimizer.step()

            step += 1
            logged_loss += loss.item() * state.shape[0]
            logged_samples += state.shape[0]
            if step % config.log_interval == 0:
                value = logged_loss / logged_samples
                if logger is None:
                    console.info(f"epoch={epoch + 1} step={step} train/loss={value:.6f}")
                else:
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
                evaluate(model, normalizer, step)
                last_evaluation_step = step

        if validation_loader is not None:
            model.eval()
            validation_loss = 0.0
            validation_samples = 0
            with torch.no_grad():
                for state, action_chunk in validation_loader:
                    state = state.to(device)
                    action_chunk = action_chunk.to(device)
                    loss = model.compute_loss(state, action_chunk)
                    validation_loss += loss.item() * state.shape[0]
                    validation_samples += state.shape[0]
            value = validation_loss / validation_samples
            if logger is None:
                console.info(f"epoch={epoch + 1} step={step} validation/loss={value:.6f}")
            else:
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
        if logger is None:
            console.info(f"epoch={config.num_epochs} step={step} train/loss={value:.6f}")
        else:
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
        evaluate(model, normalizer, step)
    return model, normalizer
