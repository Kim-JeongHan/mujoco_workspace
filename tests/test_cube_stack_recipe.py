"""Cube stacking recipe validation and expert consumption."""

from importlib.resources import files

import numpy as np
import pytest
import yaml
from controller_config import create_test_controller
from pydantic import ValidationError

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import (
    CubeStackExpert,
    CubeStackTask,
)
from mujoco_lab.behaviors.cube_stack_recipe import CubeStackRecipe
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe


def panda_recipe_data():
    path = files("mujoco_lab.behaviors").joinpath("recipes", "cube_stack", "panda.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_recipe_rejects_nonfinite_nested_values_and_invalid_ratios():
    for value in (float("nan"), float("inf"), float("-inf")):
        data = panda_recipe_data()
        data["stages"][0]["offset_xyz_m"][0] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

        data = panda_recipe_data()
        data["stages"][0]["gripper_mode"] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

        data = panda_recipe_data()
        data["stages"][3]["min_lift_height_m"] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

    for value in (0, -1, 1.1, float("nan"), float("inf")):
        data = panda_recipe_data()
        data["gripper"] = [value, 1.0]
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

        data = panda_recipe_data()
        data["stages"][0]["arm"] = [1.0, value]
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

    for value in (-1, 2, 0.5, "close"):
        data = panda_recipe_data()
        data["stages"][0]["gripper_mode"] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)


def test_changed_recipe_controls_plan_and_motion_limits():
    baseline_simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    baseline_robot = baseline_simulator.robots["panda"]
    baseline_robot.change_controller(
        create_test_controller(baseline_robot, controller="position", frame="grasp")
    )
    baseline_task = CubeStackTask(baseline_simulator, 2)
    baseline = CubeStackExpert(
        baseline_task,
        recipe=load_cube_recipe("panda"),
        method="heuristic",
    )
    original_command = baseline._plan[0].waypoints[0].copy()
    data = panda_recipe_data()
    data["stages"][0]["offset_xyz_m"][2] += 0.03
    data["arm"] = [0.1, 0.1]
    data["gripper"] = [0.125, 0.1]
    changed = CubeStackRecipe.model_validate(data)

    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["panda"]
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    expert = CubeStackExpert(task, recipe=changed, method="heuristic")
    simulator.target_updater = expert.update
    assert not np.allclose(expert._plan[0].waypoints[0][:7], original_command[:7])
    assert robot.gripper is not None
    assert robot.target is not None
    robot.gripper.set_target(0.0)
    before = robot.target.position.copy()
    simulator.run_steps(5)
    target = robot.target
    assert target is not None
    assert np.max(np.abs(target.position - before)) <= (
        max(expert.motion.motion_constraints().velocity_limit[:-1]) * 0.1 * 4 * simulator.dt + 1e-12
    )
    assert 0.0 < robot.gripper.get_target() <= 0.05 * 4 * simulator.dt


def test_grouped_cube_motion_keeps_the_strictest_member_ratios():
    data = panda_recipe_data()
    data["arm"] = [0.9, 0.8]
    data["gripper"] = [0.8, 0.9]
    for stage in data["stages"]:
        if stage["name"] == "above_place":
            stage["arm"] = [0.4, 0.7]
            stage["gripper"] = [0.5, 0.6]
        elif stage["name"] == "place":
            stage["arm"] = [0.6, 0.2]
    sim = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    robot = sim.robots["panda"]
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    expert = CubeStackExpert(
        CubeStackTask(sim, 1), recipe=CubeStackRecipe.model_validate(data), method="heuristic"
    )
    place = next(stage for stage in expert._plan if stage.recipe.name == "place")
    assert len(place.waypoints) == 3
    assert expert.stage_ratios(place) == ((0.4, 0.2), (0.5, 0.6))
    close = next(stage for stage in expert._plan if stage.recipe.name == "close")
    assert expert.stage_ratios(close) == ((0.9, 0.8), (0.8, 0.9))
