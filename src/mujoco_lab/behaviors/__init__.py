"""Physical manipulation behaviors assembled from bundled MuJoCo assets."""

from mujoco_lab.behaviors.book import BookStatus, BookTask, create_book_controller
from mujoco_lab.behaviors.book_expert import BookInsertionExpert
from mujoco_lab.behaviors.cube_stack import CubeStackTask
from mujoco_lab.behaviors.cube_stack_expert import CubeStackExpert
from mujoco_lab.behaviors.expert import Expert

__all__ = [
    "BookTask",
    "BookStatus",
    "BookInsertionExpert",
    "create_book_controller",
    "CubeStackTask",
    "CubeStackExpert",
    "Expert",
]
