"""Robot-local IK, joint limits, and isolation from shared simulation state."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.state import RobotState
from mujoco_lab.utils import Transform


@pytest.fixture
def state():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <compiler angle="radian"/>
          <default><geom size="0.02" mass="0.1" contype="0" conaffinity="0"/></default>
          <worldbody>
            <body name="object" pos="-2 0 1"><freejoint/><geom/></body>
            <body name="other/base" pos="2 0 0">
              <joint name="other/hinge" type="hinge"/><geom/>
              <site name="other/tool" pos="0.2 0 0"/>
            </body>
            <body name="marker" mocap="true" pos="0 2 1"><geom/></body>
            <body name="arm/base" pos="0.3 -0.2 0.4" quat="0.965925826 0 0 0.258819045">
              <site name="arm/base_frame"/>
              <body>
                <joint name="arm/x" type="slide" axis="1 0 0" range="-0.5 0.5"/><geom/>
                <body>
                  <joint name="arm/y" type="slide" axis="0 1 0" range="-0.5 0.5"/><geom/>
                  <body>
                    <joint name="arm/yaw" type="hinge" axis="0 0 1" limited="false"/>
                    <geom/><site name="arm/tool" pos="0.2 0 0.1"/>
                    <body pos="0 0.02 0.1">
                      <joint name="arm/finger" type="slide" axis="0 1 0" range="0 0.04"/>
                      <geom/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator><motor joint="other/hinge"/><motor joint="arm/x"/></actuator>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    result = RobotState(
        model,
        data,
        name="arm",
        prefix="arm/",
        root_name="base",
        joint_names=("x", "y", "yaw", "finger"),
        site_names=("tool", "base_frame"),
    )
    data.qpos[result.qpos_indices] = [0.03, -0.02, 0.3, 0.02]
    data.qpos[model.joint("other/hinge").qposadr[0]] = 0.4
    data.qvel[:] = np.linspace(-0.1, 0.1, model.nv)
    data.ctrl[:] = [0.1, 0.2]
    data.time = 0.37
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = np.arange(model.nv)
    return result


def pose_for(state, positions, frame="tool"):
    data = mujoco.MjData(state.model)
    mujoco.mj_copyData(data, state.model, state.data)
    data.qpos[state.qpos_indices[: len(positions)]] = positions
    mujoco.mj_kinematics(state.model, data)
    site = state.site_id(frame)
    return Transform(rotation=data.site_xmat[site].reshape(3, 3), translation=data.site_xpos[site])


def test_scalar_robot_joint_slots_ignore_earlier_environment_free_joint(state):
    assert state.nq == state.nv == 4
    assert state.qpos_indices[0] != state.dof_indices[0]
    snapshot = state.snapshot()
    np.testing.assert_array_equal(snapshot.qpos, state.data.qpos[state.qpos_indices])
    np.testing.assert_array_equal(snapshot.qvel, state.data.qvel[state.dof_indices])
    selected = snapshot.select([2, 0])
    np.testing.assert_array_equal(selected.qpos, snapshot.qpos[[2, 0]])
    np.testing.assert_array_equal(selected.qvel, snapshot.qvel[[2, 0]])


@pytest.mark.parametrize(
    "joint_xml,joint_type",
    [
        ('<freejoint name="unsupported"/>', "mjJNT_FREE"),
        ('<joint name="unsupported" type="ball"/>', "mjJNT_BALL"),
    ],
)
def test_robot_state_rejects_non_scalar_owned_joint_even_without_actuator(joint_xml, joint_type):
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco><worldbody><body name="root">
          {joint_xml}<geom type="sphere" size="0.05" mass="1"/>
        </body></worldbody></mujoco>
    """)
    with pytest.raises(ValueError, match=f"hinge/slide joints only.*unsupported.*{joint_type}"):
        RobotState(
            model,
            mujoco.MjData(model),
            name="robot",
            prefix="",
            root_name="root",
            joint_names=("unsupported",),
            site_names=(),
        )


def shared_fields(state):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(state.model, kind))
    mujoco.mj_getState(state.model, state.data, integration, kind)
    return [integration] + [
        getattr(state.data, field).copy()
        for field in ("xpos", "site_xpos", "site_xmat", "M", "qfrc_bias")
    ]


def assert_pose(actual, target):
    np.testing.assert_allclose(actual.as_translation(), target.as_translation(), atol=1e-6)
    difference = target.as_rotation() * actual.as_rotation().inv()
    assert np.linalg.norm(difference.as_rotvec()) < 1e-5


def test_ik_uses_world_pose_and_ancestor_joints_on_private_data(state, monkeypatch):
    target = pose_for(state, [0.15, -0.2, 2.6])
    before = shared_fields(state)
    untouched = np.setdiff1d(np.arange(state.model.nq), state.qpos_indices[:3])
    original_kinematics = mujoco.mj_kinematics
    evaluations = []

    def private_kinematics(model, data):
        assert data is not state.data
        np.testing.assert_array_equal(data.qpos[untouched], state.data.qpos[untouched])
        evaluations.append(data)
        original_kinematics(model, data)

    def unexpected_physics(*args, **kwargs):
        raise AssertionError("IK must not evaluate dynamics or step physics")

    monkeypatch.setattr(mujoco, "mj_kinematics", private_kinematics)
    monkeypatch.setattr(mujoco, "mj_forward", unexpected_physics)
    monkeypatch.setattr(mujoco, "mj_step", unexpected_physics)
    solution = state.solve_ik(target, frame="tool", seed=[0, 0, 2.5])
    assert solution.shape == (3,)  # The descendant finger is not optimized.
    np.testing.assert_allclose(solution, [0.15, -0.2, 2.6], atol=1e-6)
    for previous, current in zip(before, shared_fields(state)):
        np.testing.assert_array_equal(previous, current)
    assert evaluations
    assert_pose(pose_for(state, solution), target)


def test_ik_respects_limits_and_clips_the_initial_seed(state):
    yaw = state.model.joint("arm/yaw").id
    state.model.jnt_limited[yaw] = True
    state.model.jnt_range[yaw] = [-0.25, 0.25]
    target = pose_for(state, [0.2, -0.15, 0.1])
    solution = state.solve_ik(target, frame="tool", seed=[2, -2, 20])
    limits = state.model.jnt_range[state.joint_ids[:3]]
    assert np.all(solution >= limits[:, 0] + 1e-4)
    assert np.all(solution <= limits[:, 1] - 1e-4)
    assert_pose(pose_for(state, solution), target)


def test_failed_ik_leaves_shared_state_intact_and_allows_another_query(state):
    before = shared_fields(state)
    with pytest.raises(ValueError, match="Unreachable IK pose"):
        state.solve_ik(Transform(translation=[100, 100, 100]), frame="tool")
    for previous, current in zip(before, shared_fields(state)):
        np.testing.assert_array_equal(previous, current)

    state.data.qpos[state.qpos_indices[-1]] = 0.035
    target = pose_for(state, [-0.1, 0.2, -0.4])
    solution = state.solve_ik(target, frame="tool")
    assert_pose(pose_for(state, solution), target)
    assert state.data.qpos[state.qpos_indices[-1]] == 0.035


@pytest.mark.parametrize(
    "frame,message",
    [
        ("other/tool", "no site named"),
        ("base_frame", "no movable joints"),
    ],
)
def test_ik_requires_an_owned_site_with_a_movable_chain(state, frame, message):
    with pytest.raises(ValueError, match=message):
        state.solve_ik(Transform.identity(), frame=frame)


def test_forte_ik_returns_seven_arm_angles_and_keeps_the_gripper_unchanged():
    simulator = Simulator(create_environment("empty"), robots=[RobotSpec("arm", "forte")])
    state = simulator.robots["arm"].state
    state.data.qpos[state.qpos_indices[-2:]] = -0.012
    mujoco.mj_forward(state.model, state.data)
    target = pose_for(state, [0.03, -0.04, 0.03, 0.08, 0.02, 0.01, 0.04], "ee_site")
    before = shared_fields(state)
    solution = state.solve_ik(target, frame="ee_site")
    assert solution.shape == (7,)
    assert_pose(pose_for(state, solution, "ee_site"), target)
    for previous, current in zip(before, shared_fields(state)):
        np.testing.assert_array_equal(previous, current)
