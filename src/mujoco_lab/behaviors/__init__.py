"""Physical manipulation behaviors assembled from bundled MuJoCo assets."""

from mujoco_lab.behaviors.cube_stack import CubeStackTask
from mujoco_lab.behaviors.cube_stack_expert import CubeStackExpert
from mujoco_lab.behaviors.expert import Expert

__all__ = ["CubeStackTask", "CubeStackExpert", "Expert"]
