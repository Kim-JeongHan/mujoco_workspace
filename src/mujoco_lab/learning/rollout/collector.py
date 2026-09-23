"""Collect expert demonstrations through the same ``env.step`` used by policies."""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from mujoco_lab.learning.datasets.episode import Episode, save_episode
from mujoco_lab.learning.datasets.replay import (
    replay_action_repeat,
    replay_cube_yaw_range_degrees,
)
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.tasks import Expert


def _replay_metadata_for_env(
    env: gym.Env, replay_metadata: dict[str, Any] | None
) -> dict[str, Any] | None:
    if replay_metadata is None:
        return None
    result = replay_metadata.copy()
    if isinstance(env, CubeStackEnv):
        if (
            "physics_steps_per_action" in result
            and replay_action_repeat(result) != env.physics_steps_per_action
        ):
            raise ValueError("Replay action cadence differs from the environment")
        result["physics_steps_per_action"] = env.physics_steps_per_action
        if (
            "cube_yaw_range_degrees" in result
            and replay_cube_yaw_range_degrees(result) != env.cube_yaw_range_degrees
        ):
            raise ValueError("Replay cube yaw range differs from the environment")
        result["cube_yaw_range_degrees"] = env.cube_yaw_range_degrees
    return result


def collect_episode(
    env: gym.Env,
    expert: Expert,
    *,
    seed: int | None = None,
    max_steps: int = 30_000,
    options: dict[str, Any] | None = None,
    record_frame: Callable[[], dict[str, np.ndarray]] | None = None,
    replay_metadata: dict[str, Any] | None = None,
) -> Episode:
    """Collect one attempt, including failures and the final post-action state.

    Flat observations are copied for every state.
    ``info['success']`` is the only source of the success label. A missing label
    remains ``None`` even if the environment terminates.
    An optional recorder returns copied ``qpos``, scalar ``frame_times``,
    ``mocap_pos``, and ``mocap_quat`` arrays after reset and every action.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    replay_metadata = _replay_metadata_for_env(env, replay_metadata)

    obs, reset_info = env.reset(seed=seed, options=options)
    states = [obs.copy()]
    frames = [] if record_frame is None else [record_frame()]
    expert.reset(obs, dict(reset_info))
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    final_info: dict[str, Any] = {}
    collector_timeout = False

    for step_index in range(max_steps):
        action = expert.act(obs).astype(np.float32, copy=True)
        if not np.isfinite(action).all():
            raise ValueError("Expert action must be finite")
        next_obs, reward, terminated, truncated, info = env.step(action.copy())
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("Environment reward must be finite")

        expert_failed = expert.failed
        collector_timeout = step_index + 1 == max_steps and not (
            terminated or truncated or expert_failed
        )
        states.append(next_obs.copy())
        if record_frame is not None:
            frames.append(record_frame())
        actions.append(action)
        rewards.append(reward)
        terminated_flags.append(bool(terminated))
        truncated_flags.append(
            bool(truncated) or collector_timeout or (expert_failed and not terminated)
        )
        final_info = dict(info)
        obs = next_obs
        if terminated or truncated or collector_timeout or expert_failed:
            break

    success = final_info.get("success")
    if success is not None and not isinstance(success, (bool, np.bool_)):
        raise ValueError("info['success'] must be a boolean when provided")
    reason = final_info.get("termination_reason")
    if expert.failed and not (terminated or truncated):
        reason = expert.failure_reason or "expert_failed"
    elif collector_timeout:
        reason = "collector_timeout"
    elif reason is None:
        reason = "terminated" if terminated_flags[-1] else "truncated"
    metadata = {
        "seed": seed,
        "success": None if success is None else bool(success),
        "termination_reason": str(reason),
        "collector_timeout": collector_timeout,
        "expert_failed": expert.failed,
        "length": len(actions),
        "return": float(sum(rewards)),
    }
    if replay_metadata is not None:
        metadata["replay"] = replay_metadata.copy()
    if "cube_yaws_degrees" in reset_info:
        metadata["cube_yaws_degrees"] = reset_info["cube_yaws_degrees"]
    replay_arrays = (
        {
            name: np.stack([frame[name] for frame in frames])
            for name in ("qpos", "frame_times", "mocap_pos", "mocap_quat")
        }
        if frames
        else {}
    )
    return Episode(
        states=np.stack(states),
        actions=np.stack(actions),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=bool),
        truncated=np.asarray(truncated_flags, dtype=bool),
        metadata=metadata,
        **replay_arrays,
    )


_EPISODE_NAME = re.compile(r"episode_(\d{6,})\.npz")


def _resume_index(
    directory: Path,
    seed: int | None,
    replay_metadata: dict[str, Any] | None,
) -> int:
    paths = sorted(
        (int(match.group(1)), path)
        for path in directory.iterdir()
        if path.is_file() and (match := _EPISODE_NAME.fullmatch(path.name))
    )
    for expected_index, (index, path) in enumerate(paths):
        if index != expected_index:
            raise ValueError(f"Episode files have a gap before index {expected_index}")
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if not {"states.npy", "actions.npy", "metadata.npy"} <= names:
                    raise ValueError("missing required arrays")
                for member in archive.infolist():
                    with archive.open(member) as contents:
                        while contents.read(1024 * 1024):
                            pass
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
        except (OSError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Invalid episode file {path}: {exc}") from exc
        expected_seed = None if seed is None else seed + index
        if metadata.get("seed") != expected_seed:
            raise ValueError(f"Episode {path} has a different seed")
        if replay_metadata is not None:
            recorded = metadata.get("replay")
            if not isinstance(recorded, dict):
                raise ValueError(f"Episode {path} has different replay metadata")
            expected = replay_metadata.copy()
            recorded = recorded.copy()
            expected["physics_steps_per_action"] = replay_action_repeat(expected)
            recorded["physics_steps_per_action"] = replay_action_repeat(recorded)
            expected["cube_yaw_range_degrees"] = replay_cube_yaw_range_degrees(expected)
            recorded["cube_yaw_range_degrees"] = replay_cube_yaw_range_degrees(recorded)
            if recorded != expected:
                raise ValueError(f"Episode {path} has different replay metadata")
    return len(paths)


def iter_episodes(
    env: gym.Env,
    expert: Expert,
    count: int,
    *,
    seed: int | None = None,
    max_steps: int = 30_000,
    options: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    record_frame: Callable[[], dict[str, np.ndarray]] | None = None,
    replay_metadata: dict[str, Any] | None = None,
) -> Iterator[Episode]:
    """Yield each attempt after saving it, without retaining earlier episodes.

    With ``resume=True``, ``count`` is the total target, and existing episodes
    must form a valid prefix with the same seeds and replay metadata.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if resume and output_dir is None:
        raise ValueError("resume requires output_dir")
    replay_metadata = _replay_metadata_for_env(env, replay_metadata)
    directory = None if output_dir is None else Path(output_dir)
    start_index = 0
    if directory is not None:
        if resume:
            if not directory.is_dir():
                raise FileNotFoundError(f"Resume directory does not exist: {directory}")
            start_index = _resume_index(directory, seed, replay_metadata)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                path = directory / f"episode_{index:06d}.npz"
                if path.exists():
                    raise FileExistsError(path)

    for index in range(start_index, count):
        episode = collect_episode(
            env,
            expert,
            seed=None if seed is None else seed + index,
            max_steps=max_steps,
            options=options,
            record_frame=record_frame,
            replay_metadata=replay_metadata,
        )
        if directory is not None:
            save_episode(directory / f"episode_{index:06d}.npz", episode)
        yield episode
        del episode


def collect_episodes(
    env: gym.Env,
    expert: Expert,
    count: int,
    *,
    seed: int | None = None,
    max_steps: int = 30_000,
    options: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    record_frame: Callable[[], dict[str, np.ndarray]] | None = None,
    replay_metadata: dict[str, Any] | None = None,
) -> list[Episode]:
    """Collect attempts into a list for callers that need all episodes in memory."""
    return list(
        iter_episodes(
            env,
            expert,
            count,
            seed=seed,
            max_steps=max_steps,
            options=options,
            output_dir=output_dir,
            resume=resume,
            record_frame=record_frame,
            replay_metadata=replay_metadata,
        )
    )
