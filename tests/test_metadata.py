"""Typed checkpoint round trips and shared demonstration environment boundaries."""

from copy import deepcopy

import numpy as np
import pytest

from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.metadata import (
    BCSourceMetadata,
    CheckpointMetadata,
    EnvironmentMetadata,
    ObservationMetadata,
    PolicyMetadata,
    describe_episodes,
)


def metadata():
    environment = EnvironmentMetadata(
        CubeStackReplayConfig(robot="panda", cubes=1, physics_steps_per_action=5),
        dt=0.002,
        robot_name="panda",
    )
    return CheckpointMetadata(
        PolicyMetadata("gaussian", 2, 1, obs_horizon=2),
        ObservationMetadata(1),
        environment,
        algorithm="dapg",
        source_bc=BCSourceMetadata(
            PolicyMetadata("mse", 2, 1, chunk_size=3, obs_horizon=2),
            {"chunk_size": 3},
            12,
        ),
    )


def test_typed_metadata_round_trip_preserves_bc_history():
    original = metadata()
    restored = CheckpointMetadata.from_dict(original.to_dict())
    assert restored == original
    assert restored.source_bc is not None and restored.source_bc.optimizer_step == 12
    assert restored.policy.chunk_size == 1 and restored.source_bc.policy.chunk_size == 3


def test_mixed_robot_demonstrations_are_rejected_before_selecting_dataset_metadata():
    data = {
        "replay": metadata().environment.to_recording(),
        "observation": {"frame_dim": 1, "rotation_indices": []},
    }
    first = Episode(np.zeros((2, 1)), np.zeros((1, 1)), metadata=deepcopy(data))
    second = deepcopy(first)
    second.metadata["replay"].update(robot="forte", robot_name="forte")
    with pytest.raises(ValueError, match="environment metadata"):
        describe_episodes([first, second])
