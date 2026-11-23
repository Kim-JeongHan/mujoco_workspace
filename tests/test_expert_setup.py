"""Explicit task setup and expected expert failure handling."""

import gc

import pytest

from mujoco_lab import RobotSpec, Simulator, create_book_insertion, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import (
    BookInsertionExpert,
    BookTask,
    CubeStackExpert,
    CubeStackTask,
)
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout import collect_episode
from mujoco_lab.planning import default_planning
from mujoco_lab.state import IKError


@pytest.fixture(autouse=True)
def release_simulator_cycles():
    yield
    gc.collect()


def book_simulator():
    return Simulator(
        create_book_insertion(),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )


def book_task():
    simulator = book_simulator()
    return BookTask(simulator)


def test_expected_ik_failure_stops_execution_after_branch_retries(monkeypatch):
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
    assert expert.act() is None
    attempts = len(calls)
    assert attempts > 1
    assert expert.failed and expert.failure_reason == "unreachable target"
    assert expert.act() is None
    assert len(calls) == attempts


def test_task_requires_explicit_robot_setup():
    simulator = Simulator(create_book_insertion())
    with pytest.raises(ValueError, match="requires exactly one robot"):
        BookTask(simulator)


@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_cube_reset_ik_failure_is_recorded_and_next_episode_recovers(monkeypatch, method):
    simulator = Simulator(
        create_cube_stack(4),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    task = CubeStackTask(simulator, 4)
    env = CubeStackEnv(task, physics_steps_per_action=5, cube_yaw_range_degrees=45)
    original_build = CubeStackExpert.build_stages

    def unreachable(*args, **kwargs):
        raise IKError("unreachable reset pose")

    monkeypatch.setattr(CubeStackExpert, "build_stages", unreachable)
    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe("forte"),
        method=method,
        planning=default_planning() if method == "sampling" else None,
    )
    assert expert.failed
    episode = collect_episode(env, expert, seed=710, max_steps=1)
    assert len(episode) == 0
    assert simulator.data.time == 0
    assert episode.states.shape == (1, 84)
    assert episode.actions.shape == (0, 8)
    assert episode.metadata["expert_failed"] is True
    assert episode.metadata["termination_reason"] == "unreachable reset pose"
    assert expert.act() is None

    monkeypatch.setattr(CubeStackExpert, "build_stages", original_build)
    resumed = collect_episode(env, expert, seed=42, max_steps=1)
    assert len(resumed) == 1
    assert resumed.metadata["expert_failed"] is False
    assert not expert.failed
    assert expert.failure_reason is None


def test_unexpected_cube_reset_error_propagates(monkeypatch):
    simulator = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )

    def broken(*args, **kwargs):
        raise RuntimeError("planning bug")

    monkeypatch.setattr(CubeStackExpert, "build_stages", broken)
    with pytest.raises(RuntimeError, match="planning bug"):
        CubeStackExpert(
            CubeStackTask(simulator, 1),
            recipe=load_cube_recipe("forte"),
            method="sampling",
            planning=default_planning(),
        )
