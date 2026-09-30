"""Run a saved policy against one cube stacking environment."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation.progress import CubeProgressTracker
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.rendering.camera import create_free_camera
from mujoco_lab.rendering.video import VideoRecorder
from mujoco_lab.tasks.cube_stack import CubeStackTask


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
    env: CubeStackEnv,
    model: BasePolicy,
    normalizer: Normalizer,
    metadata: Mapping[str, Any],
    *,
    num_episodes: int,
    seed: int,
    policy_seed: int,
    max_steps: int,
    device: torch.device,
    flow_num_steps: int,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
    video_dir: Path | None = None,
    num_video_episodes: int = 0,
    video_fps: int = 20,
    video_width: int = 640,
    video_height: int = 480,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute fresh seeded episodes using the environment's physics timestep."""
    if seed < 0 or policy_seed < 0:
        raise ValueError("Seeds must be nonnegative")

    architecture = metadata["architecture"]
    obs_horizon = architecture["obs_horizon"]
    execution_horizon = architecture["execution_horizon"]
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
                start_sim_time = float(env.simulator.data.time)
                task = env.task
                progress = CubeProgressTracker(task) if isinstance(task, CubeStackTask) else None
                raw_obs = np.asarray(raw_obs)
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
                                clipped = bounded.astype(env.action_space.dtype, copy=False)
                                next_obs, reward, terminated, truncated, info = env.step(clipped)
                                if progress is not None:
                                    progress.observe()
                                steps += 1
                                if recorder is not None:
                                    recorder.record_due(camera)
                                episode_return += float(reward)
                                success = bool(info["success"]) and terminated and not truncated
                                raw_obs = np.asarray(next_obs)
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
                    "sim_seconds": float(env.simulator.data.time) - start_sim_time,
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
