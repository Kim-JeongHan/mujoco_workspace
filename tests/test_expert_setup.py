"""Explicit task setup and expected expert failure handling."""

import pytest

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import (
    BookInsertionExpert,
    BookTask,
)
from mujoco_lab.behaviors.book import create_book_controller
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.planning import default_planning
from mujoco_lab.state import IKError


def book_simulator():
    return Simulator(
        create_book_insertion(),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )


def book_task():
    simulator = book_simulator()
    robot = simulator.robots["forte"]
    robot.change_controller(
        create_book_controller(robot, load_robot_config(robot.robot_type).controller)
    )
    return BookTask(simulator)


def test_expected_ik_failure_is_handled_once_at_execution_boundary(monkeypatch):
    expert = BookInsertionExpert(
        book_task(),
        recipe=load_book_recipe("forte"),
        planning=default_planning(),
    )
    calls = []

    def unreachable(*args, **kwargs):
        calls.append(1)
        raise IKError("unreachable target")

    monkeypatch.setattr(expert.motion, "solve_ik", unreachable)
    expert.act()
    assert calls == [1]
    assert expert.failed and expert.failure_reason == "unreachable target"
    expert.act()
    assert calls == [1]


def test_task_requires_explicit_robot_setup():
    simulator = Simulator(create_book_insertion())
    with pytest.raises(ValueError, match="requires exactly one robot"):
        BookTask(simulator)
