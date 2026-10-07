"""Run a saved policy against one physical task environment."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_lab.behaviors.book import BookTask
from mujoco_lab.behaviors.cube_stack import CubeStackTask
from mujoco_lab.learning.config.config import RolloutConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation.progress import BookProgressTracker, CubeProgressTracker
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.timing import max_steps_for_seconds
from mujoco_lab.rendering.camera import create_free_camera
from mujoco_lab.rendering.video import VideoRecorder


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize episode outcomes, duration, and task progress."""
    successes = [row for row in episodes if row["success"]]
    summary = {
        "attempted": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "timeouts": sum(row["termination_reason"] == "time_limit" for row in episodes),
        "mean_success_sim_seconds": (
            sum(row["sim_seconds"] for row in successes) / len(successes) if successes else None
        ),
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
    if episodes and all("book_grasped" in row for row in episodes):
        for name in (
            "book_grasped",
            "book_lifted",
            "book_inserted",
            "book_released",
            "book_position_error",
            "book_rotation_error",
        ):
            summary[f"mean_{name}"] = float(np.mean([row[name] for row in episodes]))
    return summary


def evaluation_log_metrics(summary: Mapping[str, Any]) -> dict[str, int | float]:
    """Select scalar evaluation results for local and optional W&B history."""
    metrics: dict[str, int | float] = {
        "eval/attempted": summary["attempted"],
        "eval/successes": summary["successes"],
        "eval/success_rate": summary["success_rate"],
        "eval/timeouts": summary["timeouts"],
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
    for name, value in summary.items():
        if name.startswith("mean_book_"):
            metrics[f"eval/{name}"] = value
    return metrics


class PolicyEvaluator:
    """Reuse an evaluation environment and settings across policies or checkpoints.

    Episode history, progress, and recordings are local to each evaluation call.
    The caller owns the environment and chooses output paths for each run.
    """

    def __init__(self, env: Any, config: RolloutConfig, device: torch.device):
        if config.num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        self.max_steps = max_steps_for_seconds(config.max_seconds, env.action_dt)
        self.env = env
        self.config = config
        self.device = torch.device(device)

    def evaluate(
        self,
        model: BasePolicy,
        normalizer: Normalizer,
        metadata: Mapping[str, Any],
        *,
        flow_num_steps: int,
        on_episode: Callable[[dict[str, Any]], None] | None = None,
        video_dir: Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Evaluate seeded episodes and restore the model's original training mode."""
        was_training = model.training
        model.eval()
        episodes = []
        try:
            with torch.inference_mode():
                for index in range(self.config.num_episodes):
                    row = self._run_episode(
                        model, normalizer, metadata, index, flow_num_steps, video_dir
                    )
                    episodes.append(row)
                    if on_episode is not None:
                        on_episode(row)
        finally:
            model.train(was_training)
        return episodes, summarize(episodes)

    def _predict_actions(
        self,
        model: BasePolicy,
        normalizer: Normalizer,
        history: deque[np.ndarray],
        flow_num_steps: int,
    ) -> np.ndarray:
        """Normalize observation history and return physical action targets."""
        normalized = np.stack([normalizer.normalize_state(frame) for frame in history]).reshape(-1)
        state = torch.as_tensor(
            normalized.astype(np.float32, copy=False), device=self.device
        ).unsqueeze(0)
        chunk = model.sample_actions(state, num_steps=flow_num_steps)
        predicted = chunk[0].detach().cpu().numpy()
        return normalizer.denormalize_action(predicted)

    def _create_recording(
        self, index: int, seed: int, video_dir: Path | None
    ) -> tuple[AbstractContextManager[VideoRecorder | None], Any, Path | None]:
        """Prepare an optional recording; its context owns renderer cleanup."""
        if index >= self.config.video_episodes:
            return nullcontext(None), None, None
        if video_dir is None:
            raise ValueError("video_dir is required when recording evaluation episodes")
        video_path = (video_dir / f"seed_{seed}.mp4").resolve()
        if video_path.exists():
            raise FileExistsError(f"Evaluation video already exists: {video_path}")
        camera = create_free_camera(lookat=self.env.camera_lookat)
        recorder = VideoRecorder(
            self.env.simulator,
            video_path,
            fps=self.config.video_fps,
            width=self.config.video_width,
            height=self.config.video_height,
        )
        return recorder, camera, video_path

    def _run_episode(
        self,
        model: BasePolicy,
        normalizer: Normalizer,
        metadata: Mapping[str, Any],
        index: int,
        flow_num_steps: int,
        video_dir: Path | None,
    ) -> dict[str, Any]:
        """Run one fresh episode using the environment's physics timestep."""
        env = self.env
        env_seed = self.config.seed + index
        max_steps = self.max_steps
        obs_horizon = metadata["architecture"]["obs_horizon"]
        execution_horizon = metadata["architecture"]["execution_horizon"]
        low, high = env.action_space.low, env.action_space.high
        raw_obs, _ = env.reset(seed=env_seed)
        start_sim_time = float(env.simulator.data.time)
        task = env.task
        progress = (
            CubeProgressTracker(task)
            if isinstance(task, CubeStackTask)
            else BookProgressTracker(task)
            if isinstance(task, BookTask)
            else None
        )
        history = deque((raw_obs.copy() for _ in range(obs_horizon)), maxlen=obs_horizon)
        steps = 0
        episode_return = 0.0
        success = False
        reason = "time_limit"
        recording, camera, video_path = self._create_recording(index, env_seed, video_dir)
        with recording as recorder:
            if recorder is not None:
                recorder.record_initial(camera)
            with torch.random.fork_rng():
                torch.manual_seed(env_seed)
                while steps < max_steps:
                    physical = self._predict_actions(model, normalizer, history, flow_num_steps)
                    for next_act in physical[:execution_horizon]:
                        next_act = np.clip(next_act, low, high).astype(
                            env.action_space.dtype, copy=False
                        )
                        next_obs, reward, terminated, truncated, info = env.step(next_act)
                        if progress is not None:
                            progress.observe()
                        steps += 1
                        if recorder is not None:
                            recorder.record_due(camera)
                        episode_return += float(reward)
                        success = bool(info["success"]) and terminated and not truncated
                        history.append(next_obs.copy())
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
            "success": success,
            "steps": steps,
            "sim_seconds": float(env.simulator.data.time) - start_sim_time,
            "termination_reason": reason,
            "return": episode_return,
        }
        if progress is not None:
            row.update(progress.result())
        if video_path is not None:
            row["video_path"] = str(video_path)
        return row
