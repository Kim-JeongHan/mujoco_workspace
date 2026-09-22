"""Native collision checks for selected robot joints in a frozen scene."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.planning import MuJoCoCollisionChecker
from mujoco_lab.state import RobotState


def _fixture(obstacle: str = "world", *, continuous: bool = False):
    obstacles = {
        "world": '<body name="obstacle" pos="0.5 0 0"><geom type="sphere" size="0.08"/></body>',
        "other": (
            '<body name="other" pos="0 2 0">'
            '<joint name="other_joint" type="slide" axis="0 1 0"/>'
            '<geom type="sphere" pos="0.5 0 0" size="0.08"/></body>'
        ),
        "self": '<body name="sibling"><geom type="sphere" pos="0.5 0 0" size="0.08"/></body>',
        "none": "",
    }
    body = obstacles[obstacle]
    if obstacle == "self":
        body = ""
    xml = f"""
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="arm_root">
          {obstacles["self"] if obstacle == "self" else ""}
          <body name="arm_link">
            <joint name="arm_joint" type="hinge" range="-1.5 1.5"
                   limited="{"false" if continuous else "true"}"/>
            <geom name="arm_geom" type="sphere" pos="0.5 0 0" size="0.08"/>
            <site name="tool" pos="0.5 0 0"/>
            <body name="finger">
              <joint name="finger_joint" type="slide" axis="1 0 0" range="0 0.2"/>
              <geom type="sphere" pos="0.7 0 0" size="0.02" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
        {body}
        <body name="unrelated_a" pos="4 0 0"><geom type="sphere" size="0.1"/></body>
        <body name="unrelated_b" pos="4.15 0 0"><geom type="sphere" size="0.1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = RobotState(
        model,
        data,
        name="arm",
        prefix="",
        root_name="arm_root",
        joint_names=("arm_joint", "finger_joint"),
        site_names=("tool",),
    )
    return SimpleNamespace(model=model, data=data, state=state, name="arm", prefix="")


def test_edge_samples_interior_and_preserves_live_data():
    robot = _fixture()
    checker = MuJoCoCollisionChecker(robot, frame="tool")
    before = mujoco.MjData(robot.model)
    mujoco.mj_copyData(before, robot.model, robot.data)

    assert checker.joint_names == ("arm_joint",)
    assert checker.is_collision_free(np.array([-1.0]))
    assert checker.is_collision_free(np.array([1.0]))
    assert not checker.is_collision_free(np.array([0.0]))
    assert not checker.is_path_collision_free(np.array([-1.0]), np.array([1.0]), 0.1)
    coarse = MuJoCoCollisionChecker(robot, frame="tool", edge_resolution=3.0)
    assert coarse.is_path_collision_free(np.array([-1.0]), np.array([1.0]))
    np.testing.assert_array_equal(robot.data.qpos, before.qpos)
    np.testing.assert_array_equal(robot.data.qvel, before.qvel)
    np.testing.assert_array_equal(robot.data.ctrl, before.ctrl)
    assert robot.data.time == before.time


def test_unrelated_contacts_are_ignored_but_self_contacts_are_not():
    free = MuJoCoCollisionChecker(_fixture("none"), frame="tool")
    self_contact = MuJoCoCollisionChecker(_fixture("self"), frame="tool")
    assert free.is_collision_free(np.array([0.0]))
    assert not self_contact.is_collision_free(np.array([0.0]))


def test_other_body_and_finger_remain_frozen_until_refresh():
    robot = _fixture("other")
    finger = robot.model.joint("finger_joint").qposadr
    other = robot.model.joint("other_joint").qposadr
    robot.data.qpos[finger] = 0.15
    mujoco.mj_forward(robot.model, robot.data)
    checker = MuJoCoCollisionChecker(robot, frame="tool")
    assert checker.is_collision_free(np.array([0.0]))
    assert checker._scratch.qpos[finger] == 0.15

    robot.data.qpos[other] = -2.0
    mujoco.mj_forward(robot.model, robot.data)
    assert checker.is_collision_free(np.array([0.0]))
    checker.refresh()
    assert not checker.is_collision_free(np.array([0.0]))


def test_input_and_hardware_bounds_are_checked():
    robot = _fixture("none")
    with pytest.raises(ValueError, match="cannot exceed"):
        MuJoCoCollisionChecker(robot, frame="tool", bounds=[(-2.0, 2.0)])
    checker = MuJoCoCollisionChecker(robot, joint_names=["arm_joint"])
    assert not checker.is_collision_free(np.array([1.6]))
    with pytest.raises(ValueError, match="finite joint values"):
        checker.is_collision_free(np.array([np.nan]))
    with pytest.raises(ValueError, match="finite joint values"):
        checker.is_collision_free(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="resolution"):
        checker.is_path_collision_free(np.array([0.0]), np.array([0.2]), 0)
    with pytest.raises(ValueError, match="edge_resolution"):
        MuJoCoCollisionChecker(robot, frame="tool", edge_resolution=0)
    with pytest.raises(ValueError, match="explicit finite bounds"):
        MuJoCoCollisionChecker(_fixture("none", continuous=True), frame="tool")
    continuous = MuJoCoCollisionChecker(
        _fixture("none", continuous=True), frame="tool", bounds=[(-1.0, 1.0)]
    )
    assert continuous.is_collision_free(np.array([0.5]))


@pytest.mark.parametrize("robot_type", ["panda", "forte"])
def test_bundled_arm_home_and_nearby_goal(robot_type):
    simulator = Simulator(create_environment("empty"), robots=[RobotSpec(robot_type, robot_type)])
    robot = simulator.robots[robot_type]
    arm_ids = robot.state.joint_ids[:7]
    current = simulator.data.qpos[robot.model.jnt_qposadr[arm_ids]].copy()
    bounds = []
    for joint, q in zip(arm_ids, current, strict=True):
        if robot.model.jnt_limited[joint]:
            bounds.append(tuple(robot.model.jnt_range[joint]))
        else:
            bounds.append((q - 0.35, q + 0.35))
    checker = MuJoCoCollisionChecker(robot, frame="grasp", bounds=bounds)
    goal = current.copy()
    goal[1] += 0.15
    assert checker.joint_names == robot.state.joint_names[:7]
    assert checker.is_path_collision_free(current, goal)
    np.testing.assert_array_equal(simulator.data.qpos[checker.qpos_indices], current)
