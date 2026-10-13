"""Run a saved policy against one physical task environment."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mujoco_lab.behaviors.task import BookTask, CubeStackTask
from mujoco_lab.learning.config import EvaluationConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation.progress import BookProgressTracker, CubeProgressTracker
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.timing import max_steps_for_seconds
from mujoco_lab.rendering.camera import create_free_camera
from mujoco_lab.rendering.video import VideoRecorder
from mujoco_lab.simulator_manager import _passive_viewer

if TYPE_CHECKING:
    from mujoco.viewer import Handle


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
            "final_position_error",
            "grasp_fraction",
            "lift_fraction",
            "place_fraction",
        ):
            summary[f"mean_{name}"] = float(np.mean([row[name] for row in episodes]))
        summary["grasp_fraction"] = np.mean([row["grasped"] for row in episodes], axis=0).tolist()
        summary["lift_fraction"] = np.mean([row["lifted"] for row in episodes], axis=0).tolist()
        summary["place_fraction"] = np.mean([row["placed"] for row in episodes], axis=0).tolist()
    if episodes and all("inserted" in row for row in episodes):
        for name in (
            "grasped",
            "lifted",
            "inserted",
            "released",
            "position_error",
            "rotation_error",
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
        "mean_final_position_error",
        "mean_grasp_fraction",
        "mean_lift_fraction",
        "mean_place_fraction",
        "mean_grasped",
        "mean_lifted",
        "mean_inserted",
        "mean_released",
        "mean_position_error",
        "mean_rotation_error",
    ):
        if name in summary:
            metrics[f"eval/{name}"] = summary[name]
    return metrics


class PolicyEvaluator:
    """Reuse an evaluation environment and settings across policies or checkpoints.

    Episode history, progress, viewer, and recordings are local to each evaluation call.
    The caller owns the environment and chooses output paths for each run.
    Gaussian policies execute their mean; BC policies retain their sampling interface.
    """

    def __init__(self, env: Any, config: EvaluationConfig, device: torch.device):
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
        sampling_num_steps: int,
        on_episode: Callable[[dict[str, Any]], None] | None = None,
        video_dir: Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Evaluate seeded episodes and restore the model's original training mode."""
        was_training = model.training
        model.eval()
        episodes = []
        try:
            viewing = (
                _passive_viewer(
                    self.env.simulator, create_free_camera(lookat=self.env.camera_lookat)
                )
                if self.config.viewer
                else nullcontext(None)
            )
            with torch.inference_mode(), viewing as viewer:
                for index in range(self.config.num_episodes):
                    if viewer is not None and not viewer.is_running():
                        break
                    row = self._run_episode(
                        model, normalizer, metadata, index, sampling_num_steps, video_dir, viewer
                    )
                    episodes.append(row)
                    if on_episode is not None:
                        on_episode(row)
                    if row["termination_reason"] == "viewer_closed":
                        break
        finally:
            model.train(was_training)
        return episodes, summarize(episodes)

    def _predict_actions(
        self,
        model: BasePolicy,
        normalizer: Normalizer,
        history: deque[np.ndarray],
        sampling_num_steps: int,
    ) -> np.ndarray:
        """Normalize observation history and return physical action targets."""
        normalized = np.stack([normalizer.normalize_state(frame) for frame in history]).reshape(-1)
        state = torch.as_tensor(
            normalized.astype(np.float32, copy=False), device=self.device
        ).unsqueeze(0)
        chunk = (
            model(state).mean
            if isinstance(model, GaussianPolicy)
            else model.sample_actions(state, num_steps=sampling_num_steps)
        )
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
        sampling_num_steps: int,
        video_dir: Path | None,
        viewer: Handle | None = None,
    ) -> dict[str, Any]:
        """Run one fresh episode using the environment's physics timestep."""
        env = self.env
        env_seed = self.config.seed + index
        max_steps = self.max_steps
        obs_horizon = metadata["architecture"]["obs_horizon"]
        execution_horizon = metadata["architecture"]["execution_horizon"]
        with viewer.lock() if viewer is not None else nullcontext():
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
                wall_start = time.monotonic()
                while steps < max_steps:
                    if viewer is not None and not viewer.is_running():
                        reason = "viewer_closed"
                        break
                    physical = self._predict_actions(model, normalizer, history, sampling_num_steps)
                    for next_act in physical[:execution_horizon]:
                        if viewer is not None and not viewer.is_running():
                            reason = "viewer_closed"
                            break
                        with viewer.lock() if viewer is not None else nullcontext():
                            next_obs, reward, terminated, truncated, info = env.step(next_act)
                            if progress is not None:
                                progress.observe()
                            if recorder is not None:
                                recorder.record_due(camera)
                        steps += 1
                        if viewer is not None:
                            viewer.sync(state_only=True)
                            with viewer.lock():
                                env.simulator.model.opt.timestep = env.simulator.dt
                            remaining = (
                                wall_start
                                + float(env.simulator.data.time)
                                - start_sim_time
                                - time.monotonic()
                            )
                            if remaining > 0:
                                time.sleep(remaining)
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
                    if reason == "viewer_closed" or terminated or truncated or steps >= max_steps:
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
