"""Native collision probes for cube-stack planning stages."""

from types import SimpleNamespace

import mujoco
import numpy as np

from mujoco_lab.planning import CubeStackCollisionChecker, MuJoCoCollisionChecker
from mujoco_lab.state import RobotState


def _robot(
    *,
    finger_contact=False,
    arm_contact=False,
    support_height=-0.2,
    other_cube=False,
    obstacle_z=0.0,
    self_penetration=None,
):
    arm_position = "0.7 0 0" if arm_contact else "0.25 0 0"
    finger_x = 0.65 if finger_contact else 0.5
    obstacle_body = "cube1/object_0" if other_cube else "obstacle"
    obstacle_geom = "cube1/object_0" if other_cube else "obstacle_geom"
    sibling = (
        f'<body name="sibling" pos="{0.3 - self_penetration} 0 0">'
        '<geom name="sibling_geom" type="sphere" size="0.025"/></body>'
        if self_penetration is not None
        else ""
    )
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="arm_root">
          {sibling}
          <body name="arm_link">
            <joint name="arm_joint" type="hinge" range="-2 2"/>
            <geom name="arm_geom" type="sphere" pos="{arm_position}" size="0.025"/>
            <site name="grasp" pos="0.5 0 0"/>
            <body name="left_finger" pos="{finger_x} 0.065 0">
              <joint name="left_joint" type="slide" axis="0 1 0" range="-0.02 0.02"/>
              <geom name="left_pad" type="sphere" size="0.025"/>
            </body>
            <body name="right_finger" pos="{finger_x} -0.065 0">
              <joint name="right_joint" type="slide" axis="0 1 0" range="-0.02 0.02"/>
              <geom name="right_pad" type="sphere" size="0.025"/>
            </body>
          </body>
        </body>
        <body name="cube0/object_0" pos="0.7 0 0">
          <freejoint name="cube0/object_joint_0"/>
          <geom name="cube0/object_0" type="box" size="0.05 0.05 0.05"/>
        </body>
        <body name="cube0/object_target_0" pos="0.7 0 0" mocap="true"/>
        <body name="{obstacle_body}" pos="1.7 0 {obstacle_z}">
          <joint name="obstacle_joint" type="slide" axis="1 0 0" range="-2 2"/>
          <geom name="{obstacle_geom}" type="sphere" size="0.04"/>
        </body>
        <body name="table" pos="0 0 {support_height}">
          <geom name="table/box" type="plane" size="2 2 0.01"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = RobotState(
        model,
        data,
        name="arm",
        prefix="",
        root_name="arm_root",
        joint_names=("arm_joint", "left_joint", "right_joint"),
        site_names=("grasp",),
    )
    return SimpleNamespace(
        model=model, data=data, state=state, name="arm", prefix="", robot_type="fixture"
    )


def _checker(robot, stage, **kwargs):
    return CubeStackCollisionChecker(
        robot,
        stage,
        grasp_geoms=("left_pad", "right_pad"),
        **kwargs,
    )


def test_virtual_payload_hits_obstacle_inside_edge_without_moving_live_cube():
    robot = _robot()
    robot.data.qpos[robot.model.joint("obstacle_joint").qposadr] = -1.0
    mujoco.mj_forward(robot.model, robot.data)
    before = mujoco.MjData(robot.model)
    mujoco.mj_copyData(before, robot.model, robot.data)
    checker = _checker(robot, "cube0:above_place", edge_resolution=0.05)
    strict = MuJoCoCollisionChecker(robot, frame="grasp")
    start, end = np.array([-0.7]), np.array([0.7])

    assert strict.is_path_collision_free(start, end)
    assert checker.is_collision_free(start)
    assert checker.is_collision_free(end)
    assert not checker.is_path_collision_free(start, end)
    assert not checker.is_collision_free(np.array([0.0]))
    np.testing.assert_array_equal(robot.data.qpos, before.qpos)
    np.testing.assert_array_equal(robot.data.qvel, before.qvel)
    np.testing.assert_array_equal(robot.data.ctrl, before.ctrl)
    np.testing.assert_array_equal(robot.data.xpos, before.xpos)
    assert robot.data.time == before.time
    checker.is_collision_free(start)
    assert not np.array_equal(
        checker._scratch.qpos[checker._cube_qpos_adr : checker._cube_qpos_adr + 3],
        robot.data.qpos[checker._cube_qpos_adr : checker._cube_qpos_adr + 3],
    )


def test_refresh_uses_new_live_obstacle_and_cube_pose():
    robot = _robot()
    checker = _checker(robot, "cube0:above_place")
    assert checker.is_collision_free(np.array([0.0]))
    robot.data.qpos[robot.model.joint("obstacle_joint").qposadr] = -1.0
    mujoco.mj_forward(robot.model, robot.data)
    assert checker.is_collision_free(np.array([0.0]))
    checker.refresh()
    assert not checker.is_collision_free(np.array([0.0]))

    cube_adr = int(robot.model.joint("cube0/object_joint_0").qposadr[0])
    robot.data.qpos[cube_adr + 1] = 0.3
    mujoco.mj_forward(robot.model, robot.data)
    checker.refresh()
    assert checker.is_collision_free(np.array([0.0]))
    assert np.isclose(checker._relative_pos[1], 0.3)


def test_carried_cube_rejects_other_cube_contact():
    robot = _robot(other_cube=True)
    robot.data.qpos[robot.model.joint("obstacle_joint").qposadr] = -1.0
    mujoco.mj_forward(robot.model, robot.data)
    checker = _checker(robot, "cube0:above_place")
    assert not checker.is_collision_free(np.array([0.0]))


def test_place_allows_only_expected_cube_support():
    robot = _robot(other_cube=True, obstacle_z=-0.08)
    robot.data.qpos[robot.model.joint("obstacle_joint").qposadr] = -1.0
    mujoco.mj_forward(robot.model, robot.data)
    assert not _checker(robot, "cube0:place").is_collision_free(np.array([0.0]))
    supported = _checker(
        robot,
        "cube0:place",
        support_geom="cube1/object_0",
        support_penetration=0.015,
    )
    assert supported.is_collision_free(np.array([0.0]))
    assert not supported.is_collision_free(np.array([0.03]))


def test_tiny_physical_joint_limit_overshoot_does_not_widen_planning_bounds():
    robot = _robot()
    checker = _checker(robot, "cube0:above_pick")
    assert checker.bounds[0, 1] == 2.0
    assert checker.is_collision_free(np.array([2.0 + 5e-7]))
    assert not checker.is_collision_free(np.array([2.0 + 1e-4]))


def test_numerical_self_contact_is_tolerated_but_deeper_collision_is_rejected():
    shallow = _robot(self_penetration=5e-6)
    deep = _robot(self_penetration=0.005)
    shallow_checker = _checker(shallow, "cube0:above_pick")
    deep_checker = _checker(deep, "cube0:above_pick")
    assert shallow_checker.is_collision_free(np.array([0.0]))
    assert not deep_checker.is_collision_free(np.array([0.0]))


def test_grasp_allowance_is_limited_to_fingers_and_penetration():
    robot = _robot(finger_contact=True)
    close = _checker(robot, "cube0:close", grasp_penetration=0.05)
    strict = _checker(robot, "cube0:above_pick")
    assert close.is_collision_free(np.array([0.0]))
    assert not strict.is_collision_free(np.array([0.0]))
    assert not _checker(robot, "cube0:close", grasp_penetration=0.001).is_collision_free(
        np.array([0.0])
    )

    arm = _robot(finger_contact=True, arm_contact=True)
    assert not _checker(arm, "cube0:close", grasp_penetration=0.05).is_collision_free(
        np.array([0.0])
    )


def test_support_allowance_is_stage_and_depth_limited():
    robot = _robot(support_height=-0.045)
    assert _checker(robot, "cube0:lift", support_penetration=0.01).is_collision_free(
        np.array([0.0])
    )
    assert not _checker(robot, "cube0:lift", support_penetration=0.001).is_collision_free(
        np.array([0.0])
    )
    assert not _checker(robot, "cube0:above_place").is_collision_free(np.array([0.0]))
    assert _checker(
        robot, "cube0:place", support_geom="table/box", support_penetration=0.01
    ).is_collision_free(np.array([0.0]))
    # The same shallow table contact is forbidden after sweeping away from the
    # lift start or the intended placement point.
    assert not _checker(robot, "cube0:lift", support_penetration=0.01).is_collision_free(
        np.array([0.03])
    )
    assert not _checker(
        robot, "cube0:place", support_geom="table/box", support_penetration=0.01
    ).is_collision_free(np.array([0.03]))
