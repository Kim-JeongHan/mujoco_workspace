"""Physically replay recorded expert targets at a slower action cadence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from mujoco_lab.learning.datasets.episode import Episode, load_episode, save_episode
from mujoco_lab.learning.datasets.replay import capture_frame, replay_action_repeat
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.evaluation.progress import CubeProgressTracker
from mujoco_lab.utils.logger import Logger as ConsoleLogger


@dataclass
class Config:
    """Verify successful source demonstrations with physical replay."""

    data_dir: Path = Path("data/forte_heuristic_train_001")
    output_dir: Path = Path("log/diagnostics/forte_expert_100hz_replay")
    count: int = 20
    physics_steps_per_action: int = 5
    xy_range: float = 0.02
    min_gap: float = 0.01


def _source_replay(source: Episode) -> dict[str, Any]:
    metadata = source.metadata
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Source episode needs a nonnegative recorded seed")
    replay = metadata.get("replay")
    if not isinstance(replay, dict) or not replay:
        raise ValueError("Source episode needs recorded replay scene metadata")
    if replay_action_repeat(replay) != 1:
        raise ValueError("Source episode must have one physics step per recorded action")
    if replay.get("dt") != 0.002:
        raise ValueError("Source episode must have a 0.002-second physics step")
    if len(source.actions) == 0 or len(source.states) != len(source.actions) + 1:
        raise ValueError("Source episode must have T actions and T+1 observations")
    if source.frame_times is not None:
        expected = np.arange(len(source.states)) * float(replay["dt"])
        if source.frame_times.shape != expected.shape or not np.allclose(
            source.frame_times, expected, rtol=0, atol=1e-8
        ):
            raise ValueError("Source frame times do not match recorded 500 Hz actions")
    return replay


def replay_episode(
    env: Any,
    source: Episode,
    *,
    source_path: Path,
    physics_steps_per_action: int,
    replay_metadata: dict[str, Any],
) -> Episode:
    """Execute cadence-sampled targets and include the final source target."""
    if (
        isinstance(physics_steps_per_action, bool)
        or not isinstance(physics_steps_per_action, int)
        or physics_steps_per_action <= 0
    ):
        raise ValueError("physics_steps_per_action must be a positive integer")
    _source_replay(source)
    if (
        env.physics_steps_per_action != physics_steps_per_action
        or replay_action_repeat(replay_metadata) != physics_steps_per_action
    ):
        raise ValueError("Environment action repeat differs from requested replay cadence")
    seed = source.metadata["seed"]
    observation, reset_info = env.reset(seed=seed)
    if observation.shape != source.states[0].shape or not np.allclose(
        observation, source.states[0], rtol=0, atol=1e-5
    ):
        raise ValueError(f"Initial observation differs from source {source_path} seed={seed}")

    states = [observation.copy()]
    frames = [capture_frame(env.simulator)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    source_indices: list[int] = []
    drift: list[float] = []
    tracker = CubeProgressTracker(env.task)
    final_info: dict[str, Any] = {}
    source_indices_planned = list(range(0, len(source.actions), physics_steps_per_action))
    final_action_appended = source_indices_planned[-1] != len(source.actions) - 1
    if final_action_appended:
        source_indices_planned.append(len(source.actions) - 1)
    for index in source_indices_planned:
        action = source.actions[index].astype(np.float32, copy=True)
        if not np.isfinite(action).all():
            raise ValueError(f"Source {source_path} has a nonfinite action at index {index}")
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(reward):
            raise ValueError("Replay produced a nonfinite reward")
        tracker.observe()
        states.append(observation.copy())
        frames.append(capture_frame(env.simulator))
        actions.append(action)
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        source_indices.append(index)
        source_endpoint = min(index + physics_steps_per_action, len(source.actions))
        drift.append(float(np.max(np.abs(observation - source.states[source_endpoint]))))
        final_info = dict(info)
        if terminated or truncated:
            break

    success = bool(final_info.get("success") is True)
    exhausted = not (terminated_flags[-1] or truncated_flags[-1])
    if exhausted:
        truncated_flags[-1] = True
    reason = "source_exhausted" if exhausted else final_info.get("termination_reason")
    elapsed = float(frames[-1]["frame_times"] - frames[0]["frame_times"])
    source_seconds = len(source.actions) * float(source.metadata["replay"]["dt"])
    metadata = {
        "seed": seed,
        "success": success,
        "termination_reason": reason,
        "collector_timeout": False,
        "expert_failed": False,
        "length": len(actions),
        "return": float(sum(rewards)),
        "replay": replay_metadata.copy(),
        "cube_yaws_degrees": reset_info.get("cube_yaws_degrees", []),
        "source_path": str(source_path.resolve()),
        "source_seed": seed,
        "source_length": len(source.actions),
        "source_success": source.metadata.get("success"),
        "source_action_indices": source_indices,
        "source_seconds": source_seconds,
        "sim_seconds": elapsed,
        "end_time_difference_seconds": elapsed - source_seconds,
        "final_action_appended": final_action_appended,
        "final_action_executed": source_indices[-1] == len(source.actions) - 1,
        "max_observation_drift": max(drift),
        **tracker.result(),
    }
    arrays = {
        name: np.stack([frame[name] for frame in frames])
        for name in ("qpos", "frame_times", "mocap_pos", "mocap_quat")
    }
    return Episode(
        states=np.stack(states),
        actions=np.stack(actions),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=bool),
        truncated=np.asarray(truncated_flags, dtype=bool),
        metadata=metadata,
        **arrays,
    )


def run(config: Config) -> tuple[Path, dict[str, Any]]:
    """Replay the first selected successful files into a new exclusive directory."""
    console = ConsoleLogger()
    if isinstance(config.count, bool) or config.count <= 0:
        raise ValueError("count must be positive")
    if isinstance(config.physics_steps_per_action, bool) or config.physics_steps_per_action <= 0:
        raise ValueError("physics_steps_per_action must be positive")
    paths = sorted(config.data_dir.glob("episode_*.npz"))
    selected: list[Path] = []
    first: Episode | None = None
    for path in paths:
        episode = load_episode(path)
        if episode.metadata.get("success") is not True:
            continue
        _source_replay(episode)
        if first is None:
            first = episode
        elif episode.metadata["replay"] != first.metadata["replay"]:
            raise ValueError(f"Source replay scene differs in {path}")
        selected.append(path)
        if len(selected) == config.count:
            break
    if first is None or len(selected) < config.count:
        raise ValueError(f"Found only {len(selected)} successful source episodes")
    source_replay = first.metadata["replay"]
    repeat = config.physics_steps_per_action
    target_replay = {**source_replay, "physics_steps_per_action": repeat}
    wrapper = {"dataset_metadata": {"replay": target_replay}}
    max_actions = max(
        (length + repeat - 1) // repeat
        + int((length - 1) % repeat != 0)
        for length in (len(load_episode(path)) for path in selected)
    )
    env, _, _ = create_evaluation_env(
        wrapper,
        xy_range=config.xy_range,
        min_gap=config.min_gap,
        max_steps=max_actions + 1,
    )
    if not np.isclose(env.simulator.dt, source_replay["dt"], rtol=0, atol=1e-12):
        raise ValueError("Simulator physical timestep differs from source episode")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for number, path in enumerate(selected):
        source = first if number == 0 else load_episode(path)
        result = replay_episode(
            env,
            source,
            source_path=path,
            physics_steps_per_action=config.physics_steps_per_action,
            replay_metadata=target_replay,
        )
        output = config.output_dir / path.name
        save_episode(output, result)
        row = {
            "source": path.name,
            "seed": result.metadata["seed"],
            "success": result.metadata["success"],
            "reason": result.metadata["termination_reason"],
            "actions": len(result),
            "sim_seconds": result.metadata["sim_seconds"],
            "source_seconds": result.metadata["source_seconds"],
            "end_time_difference_seconds": result.metadata["end_time_difference_seconds"],
            "final_action_appended": result.metadata["final_action_appended"],
            "final_action_executed": result.metadata["final_action_executed"],
            "max_observation_drift": result.metadata["max_observation_drift"],
            "best_progress": result.metadata["best_progress"],
            "final_goal_distance": result.metadata["final_goal_distance"],
        }
        rows.append(row)
        console.info(json.dumps(row))
    summary = {
        "attempted": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "physics_steps_per_action": config.physics_steps_per_action,
        "source_data_dir": str(config.data_dir.resolve()),
        "episodes": rows,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.info(json.dumps({key: value for key, value in summary.items() if key != "episodes"}))
    return config.output_dir, summary


def main() -> None:
    run(tyro.cli(Config, description="Physically replay 500 Hz expert actions at a slower cadence"))


if __name__ == "__main__":
    main()
