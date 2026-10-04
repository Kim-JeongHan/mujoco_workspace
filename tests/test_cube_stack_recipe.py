"""Cube stacking recipe validation and expert consumption."""

import json
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


def test_editor_schema_tracks_recipe_model():
    directory = files("mujoco_lab.behaviors").joinpath("recipes", "cube_stack")
    schema = json.loads(directory.joinpath("schema.json").read_text(encoding="utf-8"))
    assert schema == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **CubeStackRecipe.model_json_schema(),
    }
    for name in ("panda", "forte"):
        content = directory.joinpath(f"{name}.yaml").read_text(encoding="utf-8")
        assert content.startswith("# yaml-language-server: $schema=./schema.json\n")
        CubeStackRecipe.model_validate(yaml.safe_load(content))


def panda_recipe_data():
    path = files("mujoco_lab.behaviors").joinpath("recipes", "cube_stack", "panda.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_recipe_rejects_invalid_workflow_and_nonfinite_limits():
    data = panda_recipe_data()
    data["stages"][0]["name"] = "release"
    with pytest.raises(ValidationError, match="ordered eight-stage"):
        CubeStackRecipe.model_validate(data)

    data = panda_recipe_data()
    data["stages"][0]["reference"] = "goal"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CubeStackRecipe.model_validate(data)

    data = panda_recipe_data()
    data["arm"]["velocity_limit"] = float("nan")
    with pytest.raises(ValidationError):
        CubeStackRecipe.model_validate(data)


def test_recipe_rejects_nonfinite_nested_values_and_nonpositive_limits():
    for value in (float("nan"), float("inf"), float("-inf")):
        data = panda_recipe_data()
        data["stages"][0]["offset_xyz_m"][0] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

        data = panda_recipe_data()
        data["stages"][0]["gripper_target_m"] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

        data = panda_recipe_data()
        data["stages"][3]["min_lift_height_m"] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

    for value in (0, -1):
        data = panda_recipe_data()
        data["gripper"] = {"velocity_limit": value}
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

        data = panda_recipe_data()
        data["stages"][0]["arm"] = {"acceleration_limit": value}
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
        recipe=load_cube_recipe(next(iter(baseline_task.simulator.robots.values())).robot_type),
        method="heuristic",
    )
    original_command = baseline._plan[0].waypoints[0].copy()
    data = panda_recipe_data()
    data["stages"][0]["offset_xyz_m"][2] += 0.03
    data["arm"] = {"velocity_limit": 0.5, "acceleration_limit": 2.0}
    data["gripper"] = {"velocity_limit": 0.05, "acceleration_limit": 0.2}
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
    expert.robot.gripper.set_target(0.0)
    before = expert.robot.target.position.copy()
    simulator.run_steps(5)
    assert np.max(np.abs(expert.robot.target.position - before)) <= (0.5 * 4 * simulator.dt + 1e-12)
    assert 0.0 < expert.robot.gripper.get_target() <= 0.05 * 4 * simulator.dt


def test_omitted_recipe_limits_inherit_robot_limits_in_grouped_stages():
    data = panda_recipe_data()
    data.pop("arm", None)
    data.pop("gripper", None)
    for stage in data["stages"]:
        stage.pop("arm", None)
        stage.pop("gripper", None)
    changed = CubeStackRecipe.model_validate(data)
    simulator = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    robot = simulator.robots["panda"]
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    expert = CubeStackExpert(CubeStackTask(simulator, 1), recipe=changed, method="heuristic")
    configured = expert.motion.motion_constraints()
    assert len(set(configured.velocity_limit)) > 1
    for stage in expert._plan:
        constraints = expert.stage_constraints(stage)
        assert constraints.velocity_limit == configured.velocity_limit
        assert constraints.acceleration_limit == configured.acceleration_limit
