"""Bookshelf recipe validation and expert consumption."""

import numpy as np
import pytest
from pydantic import ValidationError

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import book_expert
from mujoco_lab.behaviors.book import BookTask, create_book_controller
from mujoco_lab.behaviors.bookshelf_recipe import BookshelfRecipe, load_recipe
from mujoco_lab.planning import default_planning
from mujoco_lab.utils import Transform


@pytest.mark.parametrize("robot_type", ["forte", "panda"])
def test_bundled_recipes_close_to_grasp_and_transport_the_book(robot_type):
    recipe = load_recipe(robot_type)
    assert {stage.name for stage in recipe.stages if stage.gripper_mode == 1} == {
        "close",
        "lift",
        "preinsert",
        "insert",
        "lower",
    }


@pytest.mark.parametrize("pair", [[1.0], [1.0, 1.0, 1.0], {"velocity_limit": 0.5}])
@pytest.mark.parametrize("stage", [False, True])
def test_recipe_requires_velocity_acceleration_pairs(pair, stage):
    data = load_recipe("forte").model_dump()
    target = data["stages"][0] if stage else data
    target["arm"] = pair
    with pytest.raises(ValidationError):
        BookshelfRecipe.model_validate(data)


def test_recipe_defaults_to_full_ratios_and_stages_inherit_omissions():
    data = load_recipe("forte").model_dump()
    del data["arm"]
    del data["gripper"]
    del data["stages"][0]["arm"]
    recipe = BookshelfRecipe.model_validate(data)
    assert recipe.arm == recipe.gripper == (1.0, 1.0)
    assert recipe.stages[0].arm is None
    assert recipe.stages[0].gripper is None


def test_recipe_rejects_invalid_workflow_and_motion_parameters():
    data = load_recipe("forte").model_dump()
    data["stages"] = tuple(reversed(data["stages"]))
    with pytest.raises(ValidationError, match="ordered nine-stage"):
        BookshelfRecipe.model_validate(data)

    for field in ("approach_distance_m", "stage_timeout_s"):
        for value in (0, -1, float("nan"), float("inf")):
            data = load_recipe("forte").model_dump()
            data[field] = value
            with pytest.raises(ValidationError):
                BookshelfRecipe.model_validate(data)

    for index in (0, 1):
        for value in (0, -1, 1.1, float("nan"), float("inf")):
            data = load_recipe("forte").model_dump()
            data["arm"] = [1.0, 1.0]
            data["arm"][index] = value
            with pytest.raises(ValidationError):
                BookshelfRecipe.model_validate(data)

    for value in (-1, 2, 0.5, "close", float("nan"), float("inf")):
        data = load_recipe("forte").model_dump()
        data["stages"][0]["gripper_mode"] = value
        with pytest.raises(ValidationError):
            BookshelfRecipe.model_validate(data)

    data = load_recipe("forte").model_dump()
    data["min_lift_height_m"] = data["lift_height_m"]
    with pytest.raises(ValidationError, match="less than lift_height_m"):
        BookshelfRecipe.model_validate(data)

    data = load_recipe("forte").model_dump()
    data["stages"][0]["arm"] = [float("inf"), 1.0]
    with pytest.raises(ValidationError):
        BookshelfRecipe.model_validate(data)


def test_changed_recipe_controls_book_destinations_and_limits():
    data = load_recipe("forte").model_dump()
    data["approach_distance_m"] = 0.05
    data["arm"] = [0.35, 0.7]
    data["gripper"] = [0.0375, 0.06]
    data["stages"][0]["arm"] = [0.4, 0.9]
    data["stages"][0]["gripper_mode"] = 1
    data["stages"][2]["gripper"] = [0.025, 0.06]
    data["stages"][2]["gripper_mode"] = 0
    changed = BookshelfRecipe.model_validate(data)

    sim = Simulator(
        create_book_insertion(),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    robot = sim.robots["forte"]
    robot.change_controller(
        create_book_controller(robot, load_robot_config(robot.robot_type).controller)
    )
    expert = book_expert.BookInsertionExpert(
        BookTask(sim), recipe=changed, planning=default_planning()
    )
    request = expert.motion_request("approach", expert.pick_pose)
    assert isinstance(request.waypoints[-1], Transform)
    assert request.arm_ratio == (0.4, 0.9)
    assert request.gripper_ratio == (0.0375, 0.06)
    np.testing.assert_allclose(
        request.waypoints[-1].as_translation(),
        expert.pick_pose.as_translation() - expert.pick_pose.as_rotation().as_matrix()[:, 2] * 0.05,
    )
    assert request.gripper_target == 0.0
    closing = expert.motion_request("close", expert.pick_pose)
    assert closing.gripper_target == 0.074
    assert closing.arm_ratio == (0.35, 0.7)
    assert closing.gripper_ratio == (0.025, 0.06)


@pytest.mark.parametrize("robot_type", ["forte", "panda"])
def test_book_motion_requests_resolve_each_stages_gripper_mode(robot_type):
    config = load_robot_config(robot_type)
    sim = Simulator(create_book_insertion(), robots=[RobotSpec("robot", robot_type, config=config)])
    robot = sim.robots["robot"]
    robot.change_controller(create_book_controller(robot, config.controller))
    data = load_recipe(robot_type).model_dump()
    for index, stage in enumerate(data["stages"]):
        stage["gripper_mode"] = index % 2
    recipe = BookshelfRecipe.model_validate(data)
    expert = book_expert.BookInsertionExpert(BookTask(sim), recipe=recipe, method="heuristic")
    for stage in recipe.stages:
        request = expert.motion_request(stage.name, expert.pick_pose)
        limits = expert.gripper.get_control_limits()
        assert request.gripper_target == limits[0 if stage.gripper_mode == 1 else 1]
