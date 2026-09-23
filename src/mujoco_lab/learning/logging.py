"""Local JSONL learning metrics with optional Weights & Biases mirroring."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import numpy as np

from mujoco_lab.utils import Logger as ConsoleLogger


class Logger:
    """Own one fresh experiment directory and one optional W&B run.

    Each log call is an independent JSONL row. W&B history uses the actual
    optimizer step and merges calls made at the same step.
    """

    def __init__(
        self,
        run_dir: str | Path,
        config: Mapping[str, Any],
        *,
        wandb_mode: Literal["online", "disabled"] | None = None,
        wandb_project: str | None = None,
        run_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
    ) -> None:
        mode = wandb_mode if wandb_mode is not None else os.getenv("WANDB_MODE", "online")
        if mode not in ("online", "disabled"):
            raise ValueError("wandb_mode must be online or disabled")
        self.run_dir = Path(run_dir)
        self.config = json.loads(json.dumps(dict(config), allow_nan=False))
        self.wandb_mode = mode
        self.wandb_project = (
            wandb_project
            if wandb_project is not None
            else os.getenv("WANDB_PROJECT", "mujoco_workspace_bc")
        )
        self.run_name = run_name
        self.wandb_entity = wandb_entity if wandb_entity is not None else os.getenv("WANDB_ENTITY")
        self.wandb_group = wandb_group if wandb_group is not None else os.getenv("WANDB_RUN_GROUP")
        self._console = ConsoleLogger()
        self._metrics_file = None
        self._run = None
        self._last_step: int | None = None

    def __enter__(self) -> Logger:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        with (self.run_dir / "config.json").open("x", encoding="utf-8") as file:
            json.dump(self.config, file, indent=2)
            file.write("\n")
        self._metrics_file = (self.run_dir / "metrics.jsonl").open("x", encoding="utf-8")
        try:
            if self.wandb_mode != "disabled":
                import wandb

                self._run = wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    group=self.wandb_group,
                    name=self.run_name,
                    mode=self.wandb_mode,
                    force=True,
                    dir=str(self.run_dir),
                    config=self.config,
                )
                self._run.define_metric("optimizer_step")
                self._run.define_metric("*", step_metric="optimizer_step")
        except Exception:
            if self._run is not None:
                with suppress(Exception):
                    self._run.finish(exit_code=1)
            self._metrics_file.close()
            self._metrics_file = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            if self._run is not None:
                flush_error = None
                try:
                    if self._last_step is not None:
                        self._run.log({}, step=self._last_step, commit=True)
                except Exception as error:
                    flush_error = error
                try:
                    self._run.finish(exit_code=int(exc_type is not None or flush_error is not None))
                except Exception:
                    if exc_type is None and flush_error is None:
                        raise
                if exc_type is None and flush_error is not None:
                    raise flush_error
        finally:
            if self._metrics_file is not None:
                self._metrics_file.close()
                self._metrics_file = None
        return False

    def _write(self, row: dict[str, Any]) -> None:
        if self._metrics_file is None:
            raise RuntimeError("Use Logger inside its context manager")
        self._metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
        self._metrics_file.flush()

    def _check_step(self, step: int) -> None:
        if step < 0 or (self._last_step is not None and step < self._last_step):
            raise ValueError("optimizer step must be nonnegative and nondecreasing")

    def _log_run(self, payload: dict[str, Any], step: int) -> None:
        self._last_step = step
        if self._run is not None:
            self._run.log(payload, step=step, commit=False)

    def log(self, metrics: Mapping[str, int | float], *, step: int) -> None:
        """Log scalar metrics at an optimizer step without changing the caller's row."""
        self._check_step(step)
        if "optimizer_step" in metrics:
            raise ValueError("optimizer_step is reserved")
        row = {"optimizer_step": step, **metrics}
        self._write(row)
        self._console.info(json.dumps(row, allow_nan=False))
        self._log_run(row, step)

    def log_video(self, name: str, frames: np.ndarray, *, step: int, fps: int = 20) -> None:
        """Log supplied uint8 RGB frames in THWC order as a W&B video."""
        if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3:
            raise ValueError("frames must be uint8 RGB with shape (T, H, W, 3)")
        if len(frames) == 0 or fps <= 0:
            raise ValueError("video requires frames, positive fps, and a nonnegative step")
        self._check_step(step)
        self._write({"optimizer_step": step, "event": "video", "name": name, "frames": len(frames)})
        payload: dict[str, Any] = {"optimizer_step": step}
        if self._run is not None:
            import wandb

            payload[name] = wandb.Video(np.moveaxis(frames, -1, 1), fps=fps, format="mp4")
        self._log_run(payload, step)

    def log_video_file(self, name: str, path: str | Path, *, step: int) -> None:
        """Log an already finalized MP4 without loading or encoding its frames."""
        video_path = Path(path)
        self._check_step(step)
        if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
            raise ValueError("video must be an existing MP4 file and step nonnegative")
        self._write(
            {"optimizer_step": step, "event": "video", "name": name, "path": str(video_path)}
        )
        payload: dict[str, Any] = {"optimizer_step": step}
        if self._run is not None:
            import wandb

            payload[name] = wandb.Video(str(video_path), format="mp4")
        self._log_run(payload, step)

    def log_evaluation(
        self,
        metrics: Mapping[str, int | float],
        episodes: Sequence[Mapping[str, Any]],
        *,
        step: int,
    ) -> None:
        """Submit one evaluation history row with indexed recorded episodes."""
        self._check_step(step)
        if "optimizer_step" in metrics:
            raise ValueError("optimizer_step is reserved")
        videos = []
        for index, episode in enumerate(episodes):
            if "video_path" not in episode:
                continue
            path = Path(episode["video_path"])
            if not path.is_file() or path.suffix.lower() != ".mp4":
                raise ValueError("video must be an existing MP4 file")
            seed = episode["env_seed"]
            name = f"eval/rollout_ep{index}"
            caption = f"Episode {index}, env seed {seed}, optimizer step {step}"
            videos.append((name, path, seed, index, caption))
        row = {"optimizer_step": step, **metrics}
        self._write(row)
        self._console.info(json.dumps(row, allow_nan=False))
        payload: dict[str, Any] = dict(row)
        if self._run is not None and videos:
            import wandb

            for name, path, _seed, _index, caption in videos:
                payload[name] = wandb.Video(str(path), format="mp4", caption=caption)
        for name, path, seed, index, caption in videos:
            self._write(
                {
                    "optimizer_step": step,
                    "event": "video",
                    "name": name,
                    "path": str(path),
                    "env_seed": seed,
                    "episode_index": index,
                    "caption": caption,
                }
            )
        self._log_run(payload, step)

    def log_checkpoint(self, path: str | Path, *, step: int) -> None:
        """Record a saved file and mirror it as a W&B model artifact when enabled."""
        checkpoint = Path(path)
        self._check_step(step)
        if not checkpoint.is_file():
            raise ValueError("checkpoint must be an existing file and step nonnegative")
        self._write({"optimizer_step": step, "event": "checkpoint", "path": str(checkpoint)})
        if self._run is not None:
            import wandb

            artifact = wandb.Artifact(
                name=f"{self._run.id}-model", type="model", metadata={"optimizer_step": step}
            )
            artifact.add_file(str(checkpoint))
            self._run.log_artifact(artifact)
