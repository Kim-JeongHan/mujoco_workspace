"""Book checkpoint reconstruction, physical cadence, and video camera contract."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors.book import create_book_controller
from mujoco_lab.learning.config.config import RolloutConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.replay import book_metadata, model_signature
from mujoco_lab.learning.evaluate import create_evaluation_env
from mujoco_lab.learning.evaluate_book import BookEvalConfig
from mujoco_lab.learning.evaluation import PolicyEvaluator
from mujoco_lab.learning.policies.factory import build_policy


def metadata():
    config = load_robot_config("panda")
    sim = Simulator(
        create_book_insertion("small"), robots=[RobotSpec("arm", "panda", config=config)], dt=0.002
    )
    return {
        "architecture": {"physics_steps_per_action": 5, "obs_horizon": 1, "execution_horizon": 1},
        "dataset_metadata": {
            "replay": book_metadata(
                sim,
                book="small",
                robot="panda",
                physics_steps_per_action=5,
                xy_range=0.031,
                book_yaw_range_degrees=17.0,
            )
        },
    }


def test_book_builder_restores_scene_controller_and_cadence():
    recorded = metadata()
    env, scene = create_evaluation_env(recorded, xy_range=None, min_gap=0.01, max_steps=2)
    assert scene["scene"] == "book_insertion" and scene["book"] == "small"
    assert env.task.robot.name == "arm"
    assert env.xy_range == 0.031 and env.book_yaw_range_degrees == 17
    assert env.action_dt == pytest.approx(0.01)
    assert model_signature(env.simulator.model) == scene["model_sha256"]
    reference = create_book_controller(env.task.robot, load_robot_config("panda").controller)
    np.testing.assert_array_equal(env.task.robot.controller.kp, reference.kp)
    observation, _ = env.reset(seed=4)
    assert observation.shape == (48,)
    start = env.simulator.data.time
    env.step(
        np.clip(
            np.zeros(env.action_space.shape), env.action_space.low, env.action_space.high
        ).astype(env.action_space.dtype)
    )
    assert env.simulator.data.time - start == pytest.approx(0.01)
    recorded["architecture"]["physics_steps_per_action"] = 1
    with pytest.raises(ValueError, match="cadence"):
        create_evaluation_env(recorded, xy_range=None, min_gap=0.01, max_steps=2)


def test_actual_book_policy_rollout_uses_book_camera_and_diagnostics(tmp_path, monkeypatch):
    env, _ = create_evaluation_env(metadata(), xy_range=None, min_gap=0.01, max_steps=2)
    model = build_policy(
        "mse", state_dim=48, action_dim=env.action_space.shape[0], chunk_size=1, hidden_dims=()
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    normalizer = Normalizer(
        state_mean=np.zeros(48),
        state_std=np.ones(48),
        action_mean=np.zeros(env.action_space.shape[0]),
        action_std=np.ones(env.action_space.shape[0]),
    )
    lookats = []

    class Recorder:
        def __init__(self, simulator, output, **kwargs):
            self.output = Path(output)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.output.write_bytes(b"mp4")

        def record_initial(self, camera):
            lookats.append(camera.lookat.copy())

        def record_due(self, camera):
            pass

    monkeypatch.setattr("mujoco_lab.learning.evaluation.evaluator.VideoRecorder", Recorder)
    rows, summary = PolicyEvaluator(
        env, RolloutConfig(num_episodes=1, max_steps=2, video_episodes=1), torch.device("cpu")
    ).evaluate(model, normalizer, metadata(), flow_num_steps=1, video_dir=tmp_path)
    np.testing.assert_allclose(lookats[0], env.camera_lookat)
    assert rows[0]["steps"] == 2 and rows[0]["sim_seconds"] == pytest.approx(0.02)
    assert "book_position_error" in rows[0] and "cube_grasped" not in rows[0]
    assert "mean_book_position_error" in summary and Path(rows[0]["video_path"]).is_file()
    assert BookEvalConfig(checkpoint=tmp_path / "checkpoint.pt").rollout.max_steps == 18_000
