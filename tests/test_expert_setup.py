"""Explicit task setup and expected expert failure handling."""

import pytest

from mujoco_lab import RobotSpec, Simulator, create_book_insertion, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import (
    BookInsertionExpert,
    BookTask,
    CubeStackExpert,
    CubeStackTask,
    Expert,
)
from mujoco_lab.behaviors.book import create_book_controller
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control import create_controller
from mujoco_lab.planning.motion import MotionPlanner
from mujoco_lab.state import IKError


class RejectPlanner:
    name = "reject"
    seed = 7

    def plan(self, *args, **kwargs):
        return None


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


@pytest.mark.parametrize("task_name", ["book", "cube_stack"])
def test_explicit_expert_preserves_task_and_control(task_name):
    if task_name == "book":
        simulator = book_simulator()
    else:
        simulator = Simulator(
            create_cube_stack(1),
            robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
        )
    planner = RejectPlanner()
    robot = next(iter(simulator.robots.values()))
    robot.change_controller(
        create_book_controller(robot, load_robot_config(robot.robot_type).controller)
        if task_name == "book"
        else create_controller(robot, load_robot_config(robot.robot_type).controller)
    )
    task = BookTask(simulator) if task_name == "book" else CubeStackTask(simulator, 1)
    expert = (
        BookInsertionExpert(
            task,
            recipe=load_book_recipe(next(iter(task.simulator.robots.values())).robot_type),
            planner=planner,
        )
        if task_name == "book"
        else CubeStackExpert(
            task,
            recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
            method="sampling",
            planner=planner,
        )
    )
    assert isinstance(expert, Expert)
    assert isinstance(expert.motion, MotionPlanner)
    assert expert.motion.planner is planner
    assert expert.task.simulator is simulator
    assert expert.robot.controller is not None
    assert simulator.target_updater is None
    if task_name == "cube_stack":
        assert expert.task.cubes == 1
    expert.act()
    assert expert.failed
    assert "No reject route" in expert.failure_reason


def test_explicit_expert_preserves_existing_controller():
    simulator = book_simulator()
    robot = simulator.robots["forte"]
    controller = create_book_controller(robot, load_robot_config(robot.robot_type).controller)
    robot.change_controller(controller)
    BookInsertionExpert(
        BookTask(simulator),
        recipe=load_book_recipe("forte"),
        planner=RejectPlanner(),
    )
    assert robot.controller is controller


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_unexpected_planner_errors_propagate_to_application(error_type):
    class BrokenPlanner(RejectPlanner):
        def plan(self, *args, **kwargs):
            raise error_type("planner implementation error")

    expert = BookInsertionExpert(
        book_task(),
        recipe=load_book_recipe("forte"),
        planner=BrokenPlanner(),
    )
    with pytest.raises(error_type, match="planner implementation error"):
        expert.act()
    assert not expert.failed
    assert expert.failure_reason is None


def test_expected_ik_failure_is_handled_once_at_execution_boundary(monkeypatch):
    expert = BookInsertionExpert(
        book_task(),
        recipe=load_book_recipe("forte"),
        planner=RejectPlanner(),
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


def test_book_insertion_requires_explicit_planner():
    task = book_task()
    with pytest.raises(TypeError, match="planner"):
        BookInsertionExpert(
            task, recipe=load_book_recipe(next(iter(task.simulator.robots.values())).robot_type)
        )
    with pytest.raises(ValueError, match="explicit path planner"):
        BookInsertionExpert(
            task,
            recipe=load_book_recipe(next(iter(task.simulator.robots.values())).robot_type),
            planner=None,
        )
