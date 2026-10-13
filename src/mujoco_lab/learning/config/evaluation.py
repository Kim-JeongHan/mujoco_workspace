"""Physical evaluation episode, scene, and video settings."""

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Settings shared by periodic and standalone policy evaluation."""

    num_episodes: int = 3
    seed: int = 5_000
    max_seconds: float = 15.0  # Maximum simulated duration of each evaluation episode.
    xy_range: float | None = 0.02
    min_gap: float = 0.01
    cube_yaw_range_degrees: float = 45.0
    book_yaw_range_degrees: float | None = None
    viewer: bool = False  # Show live evaluation at real-time speed.
    video_episodes: int = 3
    video_fps: int = 20
    video_width: int = 640
    video_height: int = 480
