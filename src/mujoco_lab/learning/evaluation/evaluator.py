"""Run a saved policy against one cube stacking environment."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_lab.learning.checkpoint import checkpoint_action_repeat
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation.progress import CubeProgressTracker
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.rendering.camera import create_free_camera
from mujoco_lab.rendering.video import VideoRecorder
from mujoco_lab.tasks.cube_stack import CubeStackTask


def validate_contract(
    model: BasePolicy,
    normalizer: Normalizer,
    metadata: Mapping[str, Any],
    env: Any,
) -> None:
    """Reject checkpoint and environment dimensions or timing that cannot align."""
    architecture = metadata["architecture"]
    training = metadata["train_config"]
    for name in ("obs_horizon", "chunk_size", "execution_horizon", "physics_steps_per_action"):
        value = architecture[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Checkpoint {name} must be a positive integer")
        if name in training and training[name] != value:
            raise ValueError(f"Checkpoint {name} differs from train_config")
    repeat = checkpoint_action_repeat(dict(metadata))
    env_repeat = getattr(env, "physics_steps_per_action", 1)
    if env_repeat != repeat:
        raise ValueError(
            f"Environment physics_steps_per_action={env_repeat} differs from checkpoint {repeat}"
        )
    if architecture["execution_horizon"] > architecture["chunk_size"]:
        raise ValueError("execution_horizon cannot exceed chunk_size")
    for name in ("frame_dim", "state_dim", "action_dim"):
        if not isinstance(architecture[name], int) or architecture[name] <= 0:
            raise ValueError(f"Checkpoint {name} must be a positive integer")
    if (
        model.state_dim != architecture["state_dim"]
        or model.action_dim != architecture["action_dim"]
        or model.chunk_size != architecture["chunk_size"]
        or architecture["state_dim"] != architecture["frame_dim"] * architecture["obs_horizon"]
    ):
        raise ValueError("Checkpoint policy dimensions are inconsistent")
    if env.observation_space.shape != (architecture["frame_dim"],):
        raise ValueError("Environment observation dimension differs from checkpoint frame_dim")
    if env.action_space.shape != (architecture["action_dim"],):
        raise ValueError("Environment action dimension differs from checkpoint action_dim")
    for name, size in (
        ("state_mean", architecture["frame_dim"]),
        ("state_std", architecture["frame_dim"]),
        ("action_mean", architecture["action_dim"]),
        ("action_std", architecture["action_dim"]),
    ):
        array = getattr(normalizer, name)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"Checkpoint normalizer {name} must be finite with shape ({size},)")
        if name.endswith("std") and np.any(array <= 0):
            raise ValueError(f"Checkpoint normalizer {name} must be positive")
    low = np.asarray(env.action_space.low)
    high = np.asarray(env.action_space.high)
    if low.shape != (model.action_dim,) or high.shape != (model.action_dim,):
        raise ValueError("Environment action bounds have wrong dimensions")
    if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
        raise ValueError("Environment action bounds are invalid")


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize attempted episodes; clip fraction counts executed action timesteps."""
    successes = [row for row in episodes if row["success"]]
    steps = sum(row["steps"] for row in episodes)
    clipped = sum(row["clipped_actions"] for row in episodes)
    axis_counts = (
        np.sum([row["clipped_action_axes"] for row in episodes], axis=0).tolist()
        if episodes
        else []
    )
    axis_overrun = (
        np.sum([row["action_overrun_sum"] for row in episodes], axis=0).tolist() if episodes else []
    )
    axis_max_overrun = (
        np.max([row["action_overrun_max"] for row in episodes], axis=0).tolist() if episodes else []
    )
    summary = {
        "attempted": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "timeouts": sum(row["termination_reason"] == "time_limit" for row in episodes),
        "nonfinite_failures": sum(
            row["termination_reason"] == "nonfinite_action" for row in episodes
        ),
        "mean_success_sim_seconds": (
            sum(row["sim_seconds"] for row in successes) / len(successes) if successes else None
        ),
        "clipped_actions": clipped,
        "action_clip_fraction": clipped / steps if steps else 0.0,
        "clipped_action_axes": axis_counts,
        "action_clip_axis_fraction": [count / steps if steps else 0.0 for count in axis_counts],
        "action_overrun_sum": axis_overrun,
        "action_overrun_mean": [value / steps if steps else 0.0 for value in axis_overrun],
        "action_overrun_max": axis_max_overrun,
    }
    if episodes and all("best_progress" in row for row in episodes):
        for name in (
            "best_progress",
            "final_goal_distance",
            "grasp_fraction",
            "lift_fraction",
            "place_fraction",
        ):
            summary[f"mean_{name}"] = float(np.mean([row[name] for row in episodes]))
        summary["cube_grasp_fraction"] = np.mean(
            [row["cube_grasped"] for row in episodes], axis=0
        ).tolist()
        summary["cube_lift_fraction"] = np.mean(
            [row["cube_lifted"] for row in episodes], axis=0
        ).tolist()
        summary["cube_place_fraction"] = np.mean(
            [row["cube_placed"] for row in episodes], axis=0
        ).tolist()
    return summary


def evaluation_log_metrics(summary: Mapping[str, Any]) -> dict[str, int | float]:
    """Select scalar evaluation results for local and optional W&B history."""
    metrics: dict[str, int | float] = {
        "eval/attempted": summary["attempted"],
        "eval/successes": summary["successes"],
        "eval/success_rate": summary["success_rate"],
        "eval/timeouts": summary["timeouts"],
        "eval/nonfinite_failures": summary["nonfinite_failures"],
        "eval/action_clip_fraction": summary["action_clip_fraction"],
    }
    if summary["mean_success_sim_seconds"] is not None:
        metrics["eval/mean_success_sim_seconds"] = summary["mean_success_sim_seconds"]
    for name in (
        "mean_best_progress",
        "mean_final_goal_distance",
        "mean_grasp_fraction",
        "mean_lift_fraction",
        "mean_place_fraction",
    ):
        if name in summary:
            metrics[f"eval/{name}"] = summary[name]
    if "mean_best_progress" in summary:
        for index, fraction in enumerate(summary["action_clip_axis_fraction"]):
            metrics[f"eval/action_clip_axis_{index}_fraction"] = fraction
        for index, overrun in enumerate(summary["action_overrun_mean"]):
            metrics[f"eval/action_overrun_axis_{index}_mean"] = overrun
    return metrics


def evaluate_policy(
    env: Any,
    model: BasePolicy,
    normalizer: Normalizer,
    metadata: Mapping[str, Any],
    *,
    num_episodes: int,
    seed: int,
    policy_seed: int,
    max_steps: int,
    dt: float,
    device: torch.device,
    flow_num_steps: int,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
    video_dir: Path | None = None,
    num_video_episodes: int = 0,
    video_fps: int = 20,
    video_width: int = 640,
    video_height: int = 480,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute fresh seeded episodes; dt is seconds per physics tick."""
    validate_contract(model, normalizer, metadata, env)
    if num_episodes <= 0 or max_steps <= 0 or flow_num_steps <= 0:
        raise ValueError("num_episodes, max_steps, and flow_num_steps must be positive")
    if seed < 0 or policy_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Simulation dt must be finite and positive")
    simulator = getattr(env, "simulator", None)
    simulator_dt = getattr(simulator, "dt", None)
    if simulator_dt is not None and not np.isclose(simulator_dt, dt, rtol=0, atol=1e-12):
        raise ValueError("Evaluation physics dt differs from environment simulator dt")
    action_dt = getattr(env, "action_dt", None)
    if action_dt is not None and not np.isclose(
        action_dt, dt * metadata["architecture"]["physics_steps_per_action"], rtol=0, atol=1e-12
    ):
        raise ValueError("Evaluation action dt differs from environment action cadence")
    if not 0 <= num_video_episodes <= num_episodes:
        raise ValueError("num_video_episodes must be between zero and num_episodes")
    if num_video_episodes and video_dir is None:
        raise ValueError("video_dir is required when num_video_episodes is positive")

    architecture = metadata["architecture"]
    obs_horizon = architecture["obs_horizon"]
    execution_horizon = architecture["execution_horizon"]
    action_repeat = architecture["physics_steps_per_action"]
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    was_training = model.training
    model.eval()
    episodes: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for index in range(num_episodes):
                env_seed = seed + index
                episode_policy_seed = policy_seed + index
                raw_obs, _ = env.reset(seed=env_seed)
                simulator_data = getattr(getattr(env, "simulator", None), "data", None)
                start_sim_time = float(simulator_data.time) if simulator_data is not None else None
                task = getattr(env, "task", None)
                progress = CubeProgressTracker(task) if isinstance(task, CubeStackTask) else None
                raw_obs = np.asarray(raw_obs)
                if raw_obs.shape != (architecture["frame_dim"],) or not np.isfinite(raw_obs).all():
                    raise ValueError("Environment reset returned an invalid observation")
                history = deque((raw_obs.copy() for _ in range(obs_horizon)), maxlen=obs_horizon)
                steps = 0
                clipped_actions = 0
                clipped_action_axes = np.zeros(model.action_dim, dtype=np.int64)
                action_overrun_sum = np.zeros(model.action_dim, dtype=np.float64)
                action_overrun_max = np.zeros(model.action_dim, dtype=np.float64)
                episode_return = 0.0
                success = False
                reason = "time_limit"
                video_path = None
                camera = None
                recording = nullcontext(None)
                if index < num_video_episodes:
                    assert video_dir is not None
                    video_path = (video_dir / f"seed_{env_seed}.mp4").resolve()
                    if video_path.exists():
                        raise FileExistsError(f"Evaluation video already exists: {video_path}")
                    center = (env.task.starts.mean(axis=0) + env.task.goals.mean(axis=0)) / 2
                    center[2] += 0.12
                    camera = create_free_camera(lookat=center)
                    recording = VideoRecorder(
                        env.simulator,
                        video_path,
                        fps=video_fps,
                        width=video_width,
                        height=video_height,
                    )
                with recording as recorder:
                    if recorder is not None:
                        recorder.record_initial(camera)
                    with torch.random.fork_rng():
                        torch.manual_seed(episode_policy_seed)
                        while steps < max_steps:
                            normalized = np.stack(
                                [normalizer.normalize_state(frame) for frame in history]
                            ).reshape(-1)
                            state = torch.as_tensor(
                                normalized.astype(np.float32, copy=False), device=device
                            ).unsqueeze(0)
                            chunk = model.sample_actions(state, num_steps=flow_num_steps)
                            expected = (1, model.chunk_size, model.action_dim)
                            if chunk.shape != expected:
                                raise ValueError(
                                    f"Policy returned {tuple(chunk.shape)}, expected {expected}"
                                )
                            predicted = chunk[0].detach().cpu().numpy()
                            physical = normalizer.denormalize_action(predicted)
                            if not np.isfinite(predicted).all() or not np.isfinite(physical).all():
                                success = False
                                reason = "nonfinite_action"
                                break
                            for action in physical[:execution_horizon]:
                                bounded = np.clip(action, low, high)
                                overrun = np.abs(action - bounded)
                                clipped_axes = overrun > 0
                                clipped_actions += int(np.any(clipped_axes))
                                clipped_action_axes += clipped_axes
                                action_overrun_sum += overrun
                                action_overrun_max = np.maximum(action_overrun_max, overrun)
                                clipped = bounded.astype(
                                    getattr(env.action_space, "dtype", np.float32), copy=False
                                )
                                next_obs, reward, terminated, truncated, info = env.step(clipped)
                                if progress is not None:
                                    progress.observe()
                                steps += 1
                                if recorder is not None:
                                    recorder.record_due(camera)
                                episode_return += float(reward)
                                reported_success = info.get("success", False)
                                success = (
                                    isinstance(reported_success, (bool, np.bool_))
                                    and bool(reported_success)
                                    and terminated
                                    and not truncated
                                )
                                raw_obs = np.asarray(next_obs)
                                if (
                                    raw_obs.shape != (architecture["frame_dim"],)
                                    or not np.isfinite(raw_obs).all()
                                ):
                                    raise ValueError(
                                        "Environment step returned an invalid observation"
                                    )
                                history.append(raw_obs.copy())
                                if terminated or truncated or steps >= max_steps:
                                    if success:
                                        reason = "success"
                                    elif truncated:
                                        reason = "time_limit"
                                    elif terminated:
                                        reason = info.get("termination_reason") or "terminated"
                                    else:
                                        reason = "time_limit"
                                    break
                            if terminated or truncated or steps >= max_steps:
                                break
                row = {
                    "env_seed": env_seed,
                    "policy_seed": episode_policy_seed,
                    "success": success,
                    "steps": steps,
                    "sim_seconds": (
                        float(simulator_data.time) - start_sim_time
                        if simulator_data is not None and start_sim_time is not None
                        else steps * dt * action_repeat
                    ),
                    "termination_reason": reason,
                    "clipped_actions": clipped_actions,
                    "action_clip_rate": clipped_actions / steps if steps else 0.0,
                    "clipped_action_axes": clipped_action_axes.tolist(),
                    "action_clip_axis_fraction": (clipped_action_axes / steps).tolist()
                    if steps
                    else [0.0] * model.action_dim,
                    "action_overrun_sum": action_overrun_sum.tolist(),
                    "action_overrun_mean": (action_overrun_sum / steps).tolist()
                    if steps
                    else [0.0] * model.action_dim,
                    "action_overrun_max": action_overrun_max.tolist(),
                    "return": episode_return,
                }
                if progress is not None:
                    row.update(progress.result())
                if video_path is not None:
                    row["video_path"] = str(video_path)
                episodes.append(row)
                if on_episode is not None:
                    on_episode(row)
    finally:
        model.train(was_training)
    return episodes, summarize(episodes)
