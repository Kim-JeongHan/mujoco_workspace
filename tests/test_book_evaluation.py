"""Book checkpoint reconstruction, physical cadence, and video camera contract."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.learning.config.config import RolloutConfig
from mujoco_lab.learning.config.replay import BookReplayConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.replay import capture_metadata
from mujoco_lab.learning.evaluate import create_rollout_env
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
            "replay": capture_metadata(
                sim,
                BookReplayConfig(
                    book="small",
                    robot="panda",
                    physics_steps_per_action=5,
                    xy_range=0.031,
                    book_yaw_range_degrees=17.0,
                ),
            )
        },
    }


def test_actual_book_policy_rollout_uses_book_camera_and_diagnostics(tmp_path, monkeypatch):
    rollout = RolloutConfig(num_episodes=1, xy_range=None, max_seconds=0.02, video_episodes=1)
    env, _ = create_rollout_env(metadata(), rollout)
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
    rows, summary = PolicyEvaluator(env, rollout, torch.device("cpu")).evaluate(
        model, normalizer, metadata(), flow_num_steps=1, video_dir=tmp_path
    )
    np.testing.assert_allclose(lookats[0], env.camera_lookat)
    assert rows[0]["steps"] == 2 and rows[0]["sim_seconds"] == pytest.approx(0.02)
    assert "book_position_error" in rows[0] and "cube_grasped" not in rows[0]
    assert "mean_book_position_error" in summary and Path(rows[0]["video_path"]).is_file()
    assert BookEvalConfig(checkpoint=tmp_path / "checkpoint.pt").rollout.max_seconds == 180.0


@pytest.mark.parametrize("scene", ["cube_stack", "book_insertion"])
@pytest.mark.parametrize("repeat", [5, 10])
def test_rollout_env_converts_seconds_using_recorded_cadence(scene, repeat):
    recorded = metadata()
    replay = recorded["dataset_metadata"]["replay"]
    replay["physics_steps_per_action"] = repeat
    recorded["architecture"]["physics_steps_per_action"] = repeat
    if scene == "cube_stack":
        replay.update(
            scene=scene,
            environment="table_shelf",
            cubes=1,
            min_gap=0.01,
            cube_yaw_range_degrees=0.0,
        )
    env, effective = create_rollout_env(recorded, RolloutConfig(max_seconds=15.0))
    assert env.max_steps == (1500 if repeat == 5 else 750)
    assert env.max_steps * env.action_dt == pytest.approx(15.0)
    assert effective["action_execution_hz"] == pytest.approx(100.0 if repeat == 5 else 50.0)
