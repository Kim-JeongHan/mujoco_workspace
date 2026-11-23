"""Verify Forte grasp alignment and physical finger contacts."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS
from mujoco_lab.assets.loader import load_robot_config


def test_forte_extended_grasp_has_the_reference_frame():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_ASSETS["forte"].path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grasp = data.site("grasp")
    np.testing.assert_allclose(grasp.xpos, [0.9348233, -0.00484278, 0.17955984], atol=1e-6)
    np.testing.assert_allclose(
        grasp.xmat.reshape(3, 3), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=2e-5
    )


@pytest.mark.parametrize("pose", ["zero", "default"])
def test_forte_grasp_clears_and_holds_a_40_mm_cube(pose):
    config = load_robot_config("forte")
    config.pose.default = config.pose.named_poses()[pose]
    scene = create_environment("empty")
    body = scene.worldbody.add_body(name="cube", pos=[0, 0, 2])
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        mass=0.04,
        friction=[1.0, 0.05, 0.001],
    )
    sim = Simulator(scene, robots=[RobotSpec("arm", "forte", config=config)])
    model, data = sim.model, sim.data
    gripper = sim.robots["arm"].gripper
    assert gripper is not None
    np.testing.assert_allclose(gripper.get_control_limits(), [0.0, 0.048])
    grasp = data.site("arm/grasp")
    cube_address = int(model.joint("cube_joint").qposadr[0])
    data.qpos[cube_address : cube_address + 3] = grasp.xpos
    mujoco.mju_mat2Quat(data.qpos[cube_address + 3 : cube_address + 7], grasp.xmat)
    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    mujoco.mj_forward(model, data)
    assert data.ncon == 0

    initial_position = data.body("cube").xpos.copy()
    gripper.set_target(0)
    sim.run_steps(300)
    model.opt.gravity[:] = gravity
    sim.run_steps(500)

    cube = model.geom("cube_geom").id
    contacts = {
        contact.geom2 if contact.geom1 == cube else contact.geom1
        for contact in data.contact
        if cube in (contact.geom1, contact.geom2)
    }
    assert gripper.has_contact_on_all_fingers(contacts)
    assert np.linalg.norm(data.body("cube").xpos - initial_position) < 0.005
    assert not data.warning.number.any()
