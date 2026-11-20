"""Typed experiment descriptions, converted to dictionaries only at storage boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dacite import Config, from_dict

from mujoco_lab.learning.config.replay import BookReplayConfig, CubeStackReplayConfig

if TYPE_CHECKING:
    from mujoco_lab.learning.datasets.episode import Episode


@dataclass(frozen=True)
class EnvironmentMetadata:
    """One recorded scene and its physical timing, independent of a policy or dataset."""

    scene: CubeStackReplayConfig | BookReplayConfig
    dt: float
    robot_name: str
    mujoco_version: str | None = None
    model_sha256: str | None = None

    @property
    def action_dt(self) -> float:
        return self.dt * self.scene.physics_steps_per_action

    @classmethod
    def from_recording(cls, data: dict[str, Any]) -> EnvironmentMetadata:
        """Read the flat scene description stored in an episode NPZ."""
        if data["scene"] == "cube_stack":
            scene_type = CubeStackReplayConfig
        elif data["scene"] == "book_insertion":
            scene_type = BookReplayConfig
        else:
            raise ValueError(f"Unsupported recorded scene: {data['scene']}")
        # Construct directly: dacite cannot resolve BookType's PEP 695 alias.
        scene = scene_type(
            **{
                item.name: data[item.name]
                for item in fields(scene_type)
                if item.init and item.name in data
            }
        )
        return cls(
            scene,
            data["dt"],
            data["robot_name"],
            data.get("mujoco_version"),
            data.get("model_sha256"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentMetadata:
        return cls.from_recording(
            {**data["scene"], **{key: value for key, value in data.items() if key != "scene"}}
        )

    def to_recording(self) -> dict[str, Any]:
        """Keep the episode NPZ scene format compatible with recorded demonstrations."""
        return {
            **asdict(self.scene),
            "dt": self.dt,
            "robot_name": self.robot_name,
            "mujoco_version": self.mujoco_version,
            "model_sha256": self.model_sha256,
        }


@dataclass(frozen=True)
class ObservationMetadata:
    """The meaning and width of one observation frame."""

    frame_dim: int
    rotation_indices: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationMetadata:
        return cls(data["frame_dim"], tuple(data["rotation_indices"]))


@dataclass(frozen=True)
class DatasetMetadata:
    """Data provenance and split membership; scene settings live separately."""

    data_dir: str | None = None
    train_seeds: tuple[int | None, ...] | None = None
    validation_seeds: tuple[int | None, ...] = ()
    test_seeds: tuple[int | None, ...] = ()

    @classmethod
    def from_splits(
        cls,
        data_dir: Path,
        train: Sequence[Episode],
        validation: Sequence[Episode],
        test: Sequence[Episode],
    ) -> DatasetMetadata:
        return cls(
            str(data_dir),
            tuple(episode.metadata.get("seed") for episode in train),
            tuple(episode.metadata.get("seed") for episode in validation),
            tuple(episode.metadata.get("seed") for episode in test),
        )


def describe_episodes(
    episodes: Sequence[Episode],
) -> tuple[EnvironmentMetadata, ObservationMetadata]:
    """Validate the common environment and observation contract before splitting data."""
    environment = EnvironmentMetadata.from_recording(episodes[0].metadata["replay"])
    observation = ObservationMetadata.from_dict(episodes[0].metadata["observation"])
    for episode in episodes[1:]:
        other = EnvironmentMetadata.from_recording(episode.metadata["replay"])
        if other != environment:
            raise ValueError("Demonstration environment metadata must match across episodes")
        if ObservationMetadata.from_dict(episode.metadata["observation"]) != observation:
            raise ValueError("Demonstration observation metadata must match across episodes")
    return environment, observation


@dataclass(frozen=True)
class PolicyMetadata:
    """Network shape and inference settings; physical cadence belongs to the environment."""

    policy_type: Literal["mse", "flow", "diffusion", "gaussian"]
    state_dim: int
    action_dim: int
    chunk_size: int = 1
    obs_horizon: int = 1
    execution_horizon: int = 1
    hidden_dims: tuple[int, ...] = ()
    sampling_num_steps: int = 1
    bounded_actions: bool = False
    flow_time_embed_dim: int = 256
    diffusion_down_dims: tuple[int, ...] = ()
    diffusion_time_embed_dim: int = 0
    diffusion_scheduler_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyMetadata:
        return from_dict(cls, data, config=Config(cast=[tuple]))


@dataclass(frozen=True)
class BCSourceMetadata:
    """The source BC policy and training history used to initialize an RL policy."""

    policy: PolicyMetadata
    train_config: dict[str, Any]
    optimizer_step: int


@dataclass(frozen=True)
class CheckpointMetadata:
    """Shared description for BC, DAPG, and DDPG without dataset-shaped environments."""

    policy: PolicyMetadata
    observation: ObservationMetadata
    environment: EnvironmentMetadata
    dataset: DatasetMetadata | None = None
    algorithm: Literal["bc", "dapg", "ddpg"] = "bc"
    train_config: dict[str, Any] = field(default_factory=dict)
    optimizer_step: int = 0
    source_bc: BCSourceMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        """Snapshot metadata as primitive data for torch checkpoints and JSON logs."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Restore the current checkpoint description from primitive data."""
        return from_dict(
            cls,
            data,
            config=Config(
                cast=[tuple],
                type_hooks={EnvironmentMetadata: EnvironmentMetadata.from_dict},
            ),
        )
