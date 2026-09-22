"""Full Forte tabletop pickups and released, contact-supported stacks."""

import gc

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack, create_environment
from mujoco_lab.tasks.cube_stack import CubeStackTask


@pytest.mark.parametrize("cubes", [2, 3, 4])
def test_forte_picks_and_releases_stable_stack(cubes):
    simulator = create_cube_stack(cubes, robot="forte")
    task = CubeStackTask(simulator, cubes)
    model, data = simulator.model, simulator.data
    robot = simulator.robots["forte"]
    initial_qpos = data.qpos.copy()
    initial_mocap = data.mocap_pos.copy()
    simulator.run_steps(100)
    task.reset()
    np.testing.assert_allclose(data.qpos, initial_qpos)
    np.testing.assert_allclose(data.mocap_pos, initial_mocap)
    assert task.stage == 0
    assert robot.controller.gripper_target == 0

    pad_contacts = [set() for _ in range(cubes)]
    lifted_clear = np.zeros(cubes, dtype=bool)
    for _ in range(300):
        simulator.run_steps(100)
        pairs = {
            frozenset((model.geom(contact.geom1).name, model.geom(contact.geom2).name))
            for contact in data.contact
        }
        for i in range(cubes):
            cube = f"cube{i}/object_0"
            for side in ("left", "right"):
                if frozenset((cube, f"forte/gripper_{side}_pad")) in pairs:
                    pad_contacts[i].add(side)
            lifted_clear[i] |= (
                data.body(cube).xpos[2] > task.starts[i, 2] + 0.04
                and frozenset((cube, "table/box")) not in pairs
            )
        if task.status().released_stable_stack:
            break

    status = task.status()
    assert task.stage_name == "settle"
    assert status.released_stable_stack
    assert status.support_contacts
    assert np.all(status.goal_distances < 0.01)
    assert all(pads == {"left", "right"} for pads in pad_contacts)
    assert np.all(lifted_clear)
    assert not any(
        (
            model.geom(contact.geom1).name.startswith("forte/")
            and model.geom(contact.geom2).name.startswith("cube")
        )
        or (
            model.geom(contact.geom2).name.startswith("forte/")
            and model.geom(contact.geom1).name.startswith("cube")
        )
        for contact in data.contact
    )
    assert not data.warning.number.any()
    del task, simulator, robot, data, model
    gc.collect()


def test_forte_pickup_accepts_custom_robot_instance_name():
    simulator = Simulator(
        create_environment("cube_stack_2"),
        robots=[RobotSpec("arm", "forte")],
    )
    task = CubeStackTask(simulator, 2)
    simulator.run_steps(1500)
    assert task.stage_name == "cube0:lift"
    assert task.max_lift[0] > task.starts[0, 2] + 0.04
    assert not simulator.data.warning.number.any()
