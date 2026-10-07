"""Forte heuristic paths follow the rebased model without self collisions."""

import gc
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe
from mujoco_lab.control import create_controller


@pytest.fixture(autouse=True)
def release_simulator_cycles():
    yield
    gc.collect()


def make_expert(cubes=2, *, scene=None):
    config = load_robot_config("forte")
    simulator = Simulator(
        create_cube_stack(cubes) if scene is None else scene,
        robots=[RobotSpec("forte", "forte", config=config)],
    )
    robot = simulator.robots["forte"]
    robot.change_controller(create_controller(robot, config.controller))
    task = CubeStackTask(simulator, cubes)
    expert = CubeStackExpert(task, recipe=load_recipe("forte"), method="heuristic")
    return simulator, task, expert


@pytest.mark.parametrize("variant", ["forte", "forte_backup"])
@pytest.mark.parametrize("cubes", [1, 2, 3, 4])
def test_rebased_forte_heuristic_completes_a_collision_free_stack(variant, cubes, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1] / "src/mujoco_lab/assets/robot" / variant / "robot.xml"
    )
    monkeypatch.setitem(ROBOT_ASSETS, "forte", RobotAsset(path))
    simulator, task, expert = make_expert(cubes)
    self_contacts = set()
    grasp_site = expert.robot.state.site_id("grasp")
    max_tilt = 0.0

    def update(sim):
        nonlocal max_tilt
        direction = sim.data.site_xmat[grasp_site].reshape(3, 3)[:, 2]
        max_tilt = max(max_tilt, np.arccos(np.clip(-direction[2], -1, 1)))
        for contact in sim.data.contact:
            names = (sim.model.geom(contact.geom1).name, sim.model.geom(contact.geom2).name)
            if contact.dist < -1e-5 and all(name.startswith("forte/") for name in names):
                self_contacts.add(names)
        expert.update(sim)

    simulator.target_updater = update
    simulator.run_steps(30_000)
    status = task.status()
    assert not expert.failed, expert.failure_reason
    assert expert.get_stage_name() == "settle"
    assert status.released_stable_stack
    assert status.support_contacts
    assert np.all(task.max_lift > task.starts[:, 2] + 0.04)
    assert not self_contacts, self_contacts
    assert max_tilt < np.deg2rad(1), np.rad2deg(max_tilt)
    assert not simulator.data.warning.number.any()


@pytest.mark.parametrize("phase", ["close", "release"])
def test_gripper_stage_keeps_the_selected_arm_ik_branch(phase):
    _, _, expert = make_expert()
    stage = next(stage for stage in expert._plan if stage.recipe.name == phase)
    selected = expert.robot.target.position.copy()
    assert not np.allclose(selected, stage.waypoints[-1])
    trajectory, reason = expert.make_trajectory(stage)
    assert reason is None
    np.testing.assert_array_equal(trajectory.path[:, :7], np.tile(selected, (2, 1)))
    assert trajectory.path[-1, 7] == stage.gripper_target


def test_blocked_heuristic_pose_is_reported_at_execution():
    scene = create_cube_stack(2)
    scene.worldbody.add_geom(
        name="obstacle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.3, 0.5], pos=[0.4, 0, 1.0]
    )
    simulator, _, expert = make_expert(scene=scene)
    initial = simulator.data.qpos.copy()
    action = expert.act()
    assert expert.failed
    assert expert.failure_reason == "No checked heuristic route for cube0:pick"
    assert action.shape == (8,)
    np.testing.assert_array_equal(simulator.data.qpos, initial)
    assert simulator.data.time == 0
