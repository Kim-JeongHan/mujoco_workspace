"""Load demonstration episodes and split them without transition leakage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from mujoco_lab.learning.datasets.episode import Episode, load_episode
from mujoco_lab.learning.datasets.replay import replay_action_repeat


def validate_episode_cadence(episodes: Sequence[Episode], physics_steps_per_action: int) -> None:
    """Require every episode to use the configured physical timebase."""
    if (
        not isinstance(physics_steps_per_action, int)
        or isinstance(physics_steps_per_action, bool)
        or physics_steps_per_action <= 0
    ):
        raise ValueError("physics_steps_per_action must be a positive integer")
    reference_dt = None
    for index, episode in enumerate(episodes):
        replay = episode.metadata.get("replay") or {}
        repeat = replay_action_repeat(replay)
        if repeat != physics_steps_per_action:
            raise ValueError(
                f"Episode {index} physics_steps_per_action={repeat} differs from config "
                f"{physics_steps_per_action}; resample or recollect with "
                "--physics-steps-per-action matching the config"
            )
        dt = replay.get("dt")
        if dt is not None:
            if (
                not isinstance(dt, (int, float))
                or isinstance(dt, bool)
                or not np.isfinite(dt)
                or dt <= 0
            ):
                raise ValueError(f"Episode {index} replay dt must be finite and positive")
            dt = float(dt)
        if index and (
            (dt is None) != (reference_dt is None)
            or (dt is not None and not np.isclose(dt, reference_dt, rtol=0, atol=1e-12))
        ):
            raise ValueError(f"Episode {index} replay physics dt differs from other episodes")
        reference_dt = dt


def load_episodes(data_dir: str | Path, *, success_only: bool = True) -> list[Episode]:
    """Load sorted top-level NPZ episodes with a common training shape.

    By default only episodes whose metadata has ``success is True`` are kept.
    Failed and unlabeled attempts remain available with ``success_only=False``.
    Optional rewards and replay arrays are left untouched.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise ValueError(f"Episode directory does not exist: {directory}")
    paths = sorted(path for path in directory.glob("*.npz") if path.is_file())
    if not paths:
        raise ValueError(f"No NPZ episodes found in {directory}")

    episodes: list[Episode] = []
    dimensions: tuple[int, int] | None = None
    for path in paths:
        try:
            episode = load_episode(path)
            if not isinstance(episode.metadata, dict):
                raise ValueError("metadata must be an object")
            if success_only and episode.metadata.get("success") is not True:
                continue
            states, actions = episode.states, episode.actions
            for name, values in (("states", states), ("actions", actions)):
                if (
                    values.ndim != 2
                    or not np.issubdtype(values.dtype, np.number)
                    or np.issubdtype(values.dtype, np.complexfloating)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"{name} must be a finite real 2D array")
            if (
                len(actions) == 0
                or states.shape[1] == 0
                or actions.shape[1] == 0
                or len(states) != len(actions) + 1
            ):
                raise ValueError("states/actions must have shapes (T+1, S)/(T, A) with T,S,A > 0")
            shape = (states.shape[1], actions.shape[1])
            if dimensions is None:
                dimensions = shape
            elif shape != dimensions:
                raise ValueError(f"feature dimensions {shape} differ from expected {dimensions}")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, BadZipFile) as error:
            raise ValueError(f"Invalid episode {path}: {error}") from error
        episodes.append(episode)

    if not episodes:
        raise ValueError(f"No eligible episodes found in {directory}")
    return episodes


def split_episodes(
    episodes: Sequence[Episode],
    *,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Episode], list[Episode], list[Episode]]:
    """Split whole episodes while keeping repeated known seeds together.

    Unknown seeds each form their own group. Positive holdout ratios receive at
    least one group each after rounding, and one group must remain for training.
    Ratios are approximate by group, so episode counts can differ for repeated
    seeds. Within each partition, episodes retain their input order.
    """
    if not episodes:
        raise ValueError("At least one episode is required")
    if (
        not np.isfinite(validation_ratio)
        or not np.isfinite(test_ratio)
        or validation_ratio < 0
        or test_ratio < 0
        or validation_ratio + test_ratio >= 1
    ):
        raise ValueError("Holdout ratios must be finite, nonnegative, and sum to less than 1")

    group_for_episode: list[int] = []
    known_seeds: dict[object, int] = {}
    group_count = 0
    for episode in episodes:
        episode_seed = episode.metadata.get("seed")
        if episode_seed is None:
            group = group_count
            group_count += 1
        else:
            try:
                group = known_seeds.get(episode_seed)
                if group is None:
                    group = group_count
                    known_seeds[episode_seed] = group
                    group_count += 1
            except TypeError as error:
                raise ValueError("Episode seed must be hashable") from error
        group_for_episode.append(group)

    def holdout_count(ratio: float) -> int:
        return max(1, round(group_count * ratio)) if ratio > 0 else 0

    validation_count = holdout_count(validation_ratio)
    test_count = holdout_count(test_ratio)
    if validation_count + test_count >= group_count:
        raise ValueError("Not enough independent seed groups for the requested holdouts and train")

    shuffled = np.random.default_rng(seed).permutation(group_count)
    test_groups = set(shuffled[:test_count])
    validation_groups = set(shuffled[test_count : test_count + validation_count])
    train: list[Episode] = []
    validation: list[Episode] = []
    test: list[Episode] = []
    for episode, group in zip(episodes, group_for_episode, strict=True):
        if group in test_groups:
            test.append(episode)
        elif group in validation_groups:
            validation.append(episode)
        else:
            train.append(episode)
    return train, validation, test
