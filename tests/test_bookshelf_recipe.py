"""Bookshelf recipe validation and expert consumption."""

import json
from importlib.resources import files

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import book_expert
from mujoco_lab.behaviors.book import BookTask, create_book_controller
from mujoco_lab.behaviors.bookshelf_recipe import BookshelfRecipe, load_recipe
from mujoco_lab.planning import default_planning
from mujoco_lab.utils import Transform


def test_editor_schema_tracks_recipe_model():
    directory = files("mujoco_lab.behaviors").joinpath("recipes", "bookshelf")
    schema = json.loads(directory.joinpath("schema.json").read_text(encoding="utf-8"))
    assert schema == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **BookshelfRecipe.model_json_schema(),
    }
    for name in ("panda", "forte"):
        content = directory.joinpath(f"{name}.yaml").read_text(encoding="utf-8")
        assert content.startswith("# yaml-language-server: $schema=./schema.json\n")
        assert load_recipe(name) == BookshelfRecipe.model_validate(yaml.safe_load(content))


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

    for field in ("velocity_limit", "acceleration_limit"):
        for value in (0, -1, float("nan"), float("inf")):
            data = load_recipe("forte").model_dump()
            data["arm"][field] = value
            with pytest.raises(ValidationError):
                BookshelfRecipe.model_validate(data)

    data = load_recipe("forte").model_dump()
    data["min_lift_height_m"] = data["lift_height_m"]
    with pytest.raises(ValidationError, match="less than lift_height_m"):
        BookshelfRecipe.model_validate(data)

    data = load_recipe("forte").model_dump()
    data["stages"][0]["arm"]["velocity_limit"] = float("inf")
    with pytest.raises(ValidationError):
        BookshelfRecipe.model_validate(data)


def test_changed_recipe_controls_book_destinations_and_limits():
    data = load_recipe("forte").model_dump()
    data["approach_distance_m"] = 0.05
    data["arm"]["acceleration_limit"] = 0.7
    data["gripper"]["velocity_limit"] = 0.015
    data["stages"][0]["arm"]["velocity_limit"] = 0.4
    data["stages"][2]["gripper"]["velocity_limit"] = 0.01
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
    assert request.constraints is not None
    assert isinstance(request.waypoints[-1], Transform)
    np.testing.assert_allclose(request.constraints.velocity_limit, [*[0.4] * 7, 0.015])
    np.testing.assert_allclose(request.constraints.acceleration_limit, [*[0.7] * 7, 0.1])
    np.testing.assert_allclose(
        request.waypoints[-1].as_translation(),
        expert.pick_pose.as_translation() - expert.pick_pose.as_rotation().as_matrix()[:, 2] * 0.05,
    )
    assert request.gripper_target == expert.open
    closing = expert.motion_request("close", expert.pick_pose)
    assert closing.constraints is not None
    assert closing.gripper_target == expert.closed
    assert closing.constraints.velocity_limit[-1] == 0.01
    assert closing.constraints.acceleration_limit[-1] == 0.1
