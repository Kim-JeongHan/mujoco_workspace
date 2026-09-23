"""Cube stacking recipe validation and expert consumption."""

import json
from importlib.resources import files

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.control import create_controller
from mujoco_lab.tasks import (
    CubeStackExpert,
    CubeStackMotionGenerator,
    CubeStackTask,
    cube_stack_motion,
)
from mujoco_lab.tasks.cube_stack_recipe import CubeStackRecipe


def test_editor_schema_tracks_recipe_model():
    directory = files("mujoco_lab.tasks").joinpath("recipes", "cube_stack")
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
    path = files("mujoco_lab.tasks").joinpath("recipes", "cube_stack", "panda.yaml")
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
    data["arm_max_velocity"] = float("nan")
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
        data["gripper_max_velocity"] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)

        data = panda_recipe_data()
        data["stages"][0]["min_duration_s"] = value
        with pytest.raises(ValidationError):
            CubeStackRecipe.model_validate(data)


def test_changed_recipe_controls_plan_and_motion_limits(monkeypatch):
    baseline_simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("panda", "panda")])
    baseline_robot = baseline_simulator.robots["panda"]
    baseline_robot.change_controller(
        create_controller("position", baseline_robot, gravity_compensation=True, frame="grasp")
    )
    baseline_task = CubeStackTask(baseline_simulator, 2)
    baseline = CubeStackExpert(baseline_task, CubeStackMotionGenerator(baseline_task))
    original_command = baseline._plan[0].transit_commands[0].copy()
    data = panda_recipe_data()
    data["stages"][0]["offset_xyz_m"][2] += 0.03
    data["stages"][1]["min_duration_s"] = 0.202
    data["arm_max_velocity"] = 0.5
    data["arm_max_acceleration"] = 2.0
    data["gripper_max_velocity"] = 0.05
    data["gripper_max_acceleration"] = 0.2
    changed = CubeStackRecipe.model_validate(data)
    monkeypatch.setattr(cube_stack_motion, "load_recipe", lambda robot_type: changed)

    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("panda", "panda")])
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["panda"]
    robot.change_controller(
        create_controller("position", robot, gravity_compensation=True, frame="grasp")
    )
    expert = CubeStackExpert(task, CubeStackMotionGenerator(task))
    simulator.target_updater = expert.update
    assert expert._plan[0].recipe.min_duration_s == 0.202
    assert not np.allclose(expert._plan[0].transit_commands[0][:7], original_command[:7])
    expert.robot.gripper.set_target(0.0)
    before = expert.robot.target.position.copy()
    simulator.run_steps(5)
    assert np.max(np.abs(expert.robot.target.position - before)) <= (0.5 * 4 * simulator.dt + 1e-12)
    assert 0.0 < expert.robot.gripper.get_target() <= 0.05 * 4 * simulator.dt
    assert simulator.data.time - expert._stage_start_time < expert._plan[0].recipe.min_duration_s
