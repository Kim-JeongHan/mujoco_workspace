"""Runtime checks for the simplified ForteV1 RobStride model."""

import gc
from pathlib import Path

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import ENVIRONMENT_NAMES, ControlTarget, RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.forte_coordinates import HOME_DEGREES, from_legacy
from mujoco_lab.control.trajectory import PD_WAYPOINTS

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/mujoco_lab/assets/robot/forte/robot.xml"
ARM_JOINTS = (
    "shoulder_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_pitch",
    "lower_arm_roll",
    "wrist_pitch",
    "wrist_roll",
)


@pytest.fixture(autouse=True)
def release_native_models():
    """Release simulator cycles before compiling the next detailed CAD model."""
    yield
    gc.collect()


def forte_sim(environment="empty"):
    return Simulator(
        create_environment(environment),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )


def test_cad_model_has_seven_arm_axes_and_coupled_sliders():
    sim = forte_sim()
    model, data = sim.model, sim.data
    joints = ARM_JOINTS + ("gripper_left_joint", "gripper_right_joint")
    assert (model.nq, model.nv, model.nu, model.neq) == (9, 9, 8, 1)
    assert [model.joint(i).name for i in range(model.njnt)] == [f"forte/{name}" for name in joints]
    assert [model.actuator(i).name for i in range(model.nu)] == [
        *(f"forte/{name}_motor" for name in ARM_JOINTS),
        "forte/gripper_motor",
    ]
    np.testing.assert_allclose(model.actuator_ctrlrange[-1], [-0.037, 0])
    np.testing.assert_allclose(model.jnt_range[-2:], [[-0.037, 0], [-0.037, 0]])
    np.testing.assert_allclose(model.dof_armature[:7], 0.01)
    np.testing.assert_allclose(model.dof_damping[:7], 0.05)
    np.testing.assert_array_equal(data.qpos, model.key("forte/home").qpos)
    np.testing.assert_allclose(
        np.degrees(data.qpos[:7]),
        HOME_DEGREES,
        atol=1e-8,
    )
    np.testing.assert_array_equal(data.qpos[-2:], 0)
    assert model.jnt_range[1, 1] == 0
    assert data.ncon == 0
    assert not data.warning.number.any()


def test_cad_mass_and_home_pose_tool_frame():
    sim = forte_sim()
    model, data = sim.model, sim.data
    assert model.body_mass.sum() == pytest.approx(7.4975378572836, abs=1e-9)
    assert np.all(model.body_inertia[1:] > 0)
    np.testing.assert_allclose(
        data.site("forte/ee_site").xpos,
        [0.37701991, -0.00021759, 0.31030735],
        atol=1e-6,
    )
    upper_arm = data.body("forte/elbowlink").xpos - data.body("forte/upperarmright").xpos
    forearm = data.body("forte/spiral_gear_2").xpos - data.body("forte/elbowlink").xpos
    assert abs(upper_arm[0]) < 0.005
    assert upper_arm[2] > 0.33
    assert np.linalg.norm(forearm[:2]) > 0.36
    # The CAD wrist pivot is offset from the forearm centerline. Check the
    # forearm roll axis instead of forcing the wrist pivot onto that line.
    forearm_axis = data.xaxis[model.joint("forte/lower_arm_roll").id]
    assert forearm_axis[0] > 0.98
    hand = data.site("forte/ee_site").xpos - data.body("forte/part_8_2").xpos
    jaw = data.geom("forte/gripper_right_pad").xpos - data.geom("forte/gripper_left_pad").xpos
    assert hand[2] < -0.16
    assert np.linalg.norm(jaw) > 0.07
    np.testing.assert_allclose(
        model.body_quat[model.body("forte/main_drum").id], [1, 0, 0, 0], atol=1e-6
    )
    assert model.nmesh == 270
    assert sum((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (model.geom_group == 1)) == 830
    assert model.geom("forte/gripper_left_pad").id >= 0
    assert model.geom("forte/gripper_right_pad").id >= 0
    left = data.geom("forte/gripper_left_pad")
    right = data.geom("forte/gripper_right_pad")
    grasp = data.site("forte/grasp")
    pad_axes = left.xmat.reshape(3, 3)
    grasp_axes = grasp.xmat.reshape(3, 3)
    np.testing.assert_allclose(
        grasp.xpos,
        (left.xpos + right.xpos) / 2 + 0.01 * pad_axes[:, 1],
        atol=1e-6,
    )
    np.testing.assert_allclose(grasp_axes[:, 0], pad_axes[:, 0], atol=1e-6)
    np.testing.assert_allclose(grasp_axes[:, 2], pad_axes[:, 1], atol=1e-6)


def test_gripper_closes_and_reopens_under_native_physics():
    sim = forte_sim()
    robot = sim.robots["forte"]
    controller = create_test_controller(robot, controller="pd")
    robot.change_controller(controller)
    robot.gripper.set_target(-0.02)
    sim.run_steps(500)
    np.testing.assert_allclose(sim.data.qpos[-2:], -0.02, atol=0.001)
    robot.gripper.set_target(0)
    sim.run_steps(500)
    np.testing.assert_allclose(sim.data.qpos[-2:], 0, atol=0.001)
    assert not sim.data.warning.number.any()
    sim.reset()
    assert robot.gripper.get_target() == 0
    np.testing.assert_array_equal(sim.data.qpos[-2:], 0)


def test_collision_is_clear_along_the_pd_demo_with_open_and_closed_fingers():
    model = mujoco.MjModel.from_xml_path(str(RUNTIME))
    data = mujoco.MjData(model)
    for index, start in enumerate(PD_WAYPOINTS):
        end = PD_WAYPOINTS[(index + 1) % len(PD_WAYPOINTS)]
        for fraction in np.linspace(0, 1, 21):
            data.qpos[:7] = start + fraction * (end - start)
            for grip in (0, -0.02):
                data.qpos[-2:] = grip
                mujoco.mj_forward(model, data)
                assert all(contact.dist >= -0.0001 for contact in data.contact)


def test_collision_covers_previously_missed_cad_surfaces():
    """Probe the base, shoulder, palm and finger gaps found in the CAD audit."""
    spec = mujoco.MjSpec.from_file(str(RUNTIME))
    body = spec.worldbody.add_body(name="probe", pos=[0, 0, 2])
    body.add_freejoint()
    body.add_geom(
        name="probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.001, 0, 0],
        mass=0.01,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    # Restore CAD joint coordinates using physical home offsets.
    data.qpos[:7] = from_legacy(
        [
            -0.031713138037,
            0.392582419038,
            0.354095618670,
            0.005483411844,
            0.032901666522,
            -2.413011791532,
            -0.054828117868,
        ]
    )
    probe_id = model.geom("probe_geom").id
    collisions = np.flatnonzero(model.geom_group == 3)
    # Audit points are component-local so corrected parent assemblies do not
    # invalidate the original surface-coverage probes.
    for body_name, local_point in (
        ("base_link", [-0.194757400, 0.024474315, -0.058]),
        ("main_drum", [-0.0197327417651, -0.103500035554, 0.096765961]),
        ("part_8_2", [0.0176653257297, 0.0635854552359, -0.0387323109689]),
        ("part_1_35", [-0.0250382557277, 0.00108207060191, -0.00703996909981]),
        ("part_8_2", [0.00659171032368, 0.0862013055883, -0.00551563727651]),
    ):
        mujoco.mj_forward(model, data)
        anchor = data.body(body_name)
        data.qpos[-7:-4] = anchor.xpos + anchor.xmat.reshape(3, 3) @ local_point
        mujoco.mj_forward(model, data)
        clearance = min(
            mujoco.mj_geomDistance(model, data, probe_id, int(geom), 1, None) for geom in collisions
        )
        assert clearance + 0.001 < 0.005


def test_gripper_holds_lifts_and_releases_a_cube_under_gravity():
    """Start in a grasp and use real arm control, contacts and gravity throughout."""
    scene = create_environment("empty")
    cube = scene.worldbody.add_body(name="cube", pos=[1, 0, 1])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        mass=0.07936,
        friction=[1, 0.02, 0.001],
    )
    sim = Simulator(scene, robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))])
    model, data = sim.model, sim.data
    robot = sim.robots["forte"]

    def grasp_center():
        left, right = data.geom("forte/gripper_left_pad"), data.geom("forte/gripper_right_pad")
        return (left.xpos + right.xpos) / 2 + left.xmat.reshape(3, 3)[:, 1] * 0.01

    def cube_contacts():
        pairs = [
            {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
            for contact in data.contact
        ]
        return set().union(*(pair for pair in pairs if "cube_geom" in pair))

    # Place a free cube in the initial grasp; it is never repositioned after this.
    address = model.joint("cube_joint").qposadr[0]
    data.qpos[address : address + 3] = grasp_center()
    mujoco.mju_mat2Quat(
        data.qpos[address + 3 : address + 7],
        data.geom("forte/gripper_left_pad").xmat,
    )
    data.qpos[robot.state.qpos_indices[-2:]] = -0.01771
    mujoco.mj_forward(model, data)
    controller = create_test_controller(robot, controller="pd")
    robot.change_controller(controller)
    robot.gripper.set_target(-0.02)
    pads = {"forte/gripper_left_pad", "forte/gripper_right_pad"}
    assert model.neq == 1  # Only the finger coupling; no object weld.
    for _ in range(10):
        sim.run_steps(500)
        mujoco.mj_forward(model, data)
        assert pads <= cube_contacts()
        assert np.linalg.norm(data.body("cube").xpos - grasp_center()) < 0.001

    start_height = data.body("cube").xpos[2]
    robot.update_state()
    start_target = robot.target.position.copy()
    jacobian = robot.state.get_jacobian("ee_site")[:3, :7]
    delta = np.linalg.pinv(jacobian) @ np.array([0, 0, 0.05])
    for fraction in np.linspace(0, 1, 2500):
        blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        robot.target = ControlTarget(start_target + blend * delta)
        sim.physics_step()
    mujoco.mj_forward(model, data)
    assert data.body("cube").xpos[2] > start_height + 0.04
    assert pads <= cube_contacts()
    assert np.linalg.norm(data.body("cube").xpos - grasp_center()) < 0.002

    robot.gripper.set_target(0)
    sim.run_steps(1000)
    mujoco.mj_forward(model, data)
    assert "floor" in cube_contacts()
    assert not pads & cube_contacts()
    assert data.body("cube").xpos[2] < 0.03
    assert not data.warning.number.any()


@pytest.mark.parametrize("environment", ENVIRONMENT_NAMES)
def test_zero_torque_fall_is_finite(environment):
    sim = forte_sim(environment)
    model, data = sim.model, sim.data
    assert all(
        not model.geom(contact.geom1).name.startswith("forte/")
        and not model.geom(contact.geom2).name.startswith("forte/")
        for contact in data.contact
    )
    mujoco.mj_step(model, data, nstep=1000)
    assert data.time == pytest.approx(2)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert not data.warning.number.any()
