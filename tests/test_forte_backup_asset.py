"""Physical CAD finger grasp checks for both Forte collision models."""

import gc
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lab import ControlTarget, RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors.book import create_book_controller

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["forte", "forte_backup"], autouse=True)
def use_forte_asset(request, monkeypatch):
    """Run the same physical grasp with proxy and detailed CAD arm collisions."""
    path = ROOT / "src/mujoco_lab/assets/robot" / request.param / "robot.xml"
    monkeypatch.setitem(ROBOT_ASSETS, "forte", RobotAsset(path))


@pytest.fixture(autouse=True)
def release_native_models():
    """Release simulator cycles before compiling the next detailed CAD model."""
    yield
    gc.collect()


@pytest.mark.parametrize(
    ("dimensions", "mass"),
    [((0.04, 0.04, 0.04), 0.07936), ((0.034, 0.236, 0.156), 0.45), ((0.05, 0.236, 0.156), 0.65)],
    ids=["cube", "medium_book", "thick_book"],
)
def test_fingers_hold_lift_and_release_free_boxes_under_gravity(dimensions, mass):
    """Hold cube and book shapes with native CAD finger contacts and payload PD control."""
    scene = create_environment("empty")
    cube = scene.worldbody.add_body(name="cube", pos=[1, 0, 1])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.asarray(dimensions) / 2,
        mass=mass,
        friction=[1, 0.02, 0.001],
    )
    sim = Simulator(scene, robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))])
    model, data = sim.model, sim.data
    robot = sim.robots["forte"]
    gripper = robot.gripper
    assert gripper is not None
    site = model.site("forte/grasp").id
    depth_offset = dimensions[2] / 2 - 0.012 if dimensions[1] > 0.04 else 0

    def grasp_center():
        return data.site_xpos[site] + data.site_xmat[site].reshape(3, 3)[:, 2] * depth_offset

    def cube_contacts():
        cube_geom = model.geom("cube_geom").id
        return {
            int(contact.geom2 if contact.geom1 == cube_geom else contact.geom1)
            for contact in data.contact
            if cube_geom in (contact.geom1, contact.geom2) and contact.dist <= 0
        }

    # Place a free box in the initial grasp; it is never repositioned after this.
    gripper.apply_initial_width(dimensions[0] + 0.004)
    mujoco.mj_forward(model, data)
    address = int(model.joint("cube_joint").qposadr[0])
    data.qpos[address : address + 3] = grasp_center()
    mujoco.mju_mat2Quat(
        data.qpos[address + 3 : address + 7],
        data.site_xmat[site],
    )
    mujoco.mj_forward(model, data)
    controller = create_book_controller(robot, load_robot_config("forte").controller)
    robot.change_controller(controller)
    # Prepare the grasp without gravity, then hold and lift the free box under gravity.
    model.opt.gravity[:] = 0
    gripper.set_target(0)
    sim.run_steps(250)
    mujoco.mj_forward(model, data)
    grasp_offset = data.site_xmat[site].reshape(3, 3).T @ (
        data.body("cube").xpos - data.site_xpos[site]
    )
    tolerance = 0.002 if dimensions[1] == 0.04 else 0.005

    def grasp_drift():
        relative = data.site_xmat[site].reshape(3, 3).T @ (
            data.body("cube").xpos - data.site_xpos[site]
        )
        return np.linalg.norm(relative - grasp_offset)

    model.opt.gravity[:] = [0, 0, -9.81]
    assert model.neq == 1  # Only the finger coupling; no object weld.
    assert all("_pad" not in model.geom(geom).name for geom in range(model.ngeom))
    for _ in range(10):
        sim.run_steps(500)
        mujoco.mj_forward(model, data)
        assert gripper.has_contact_on_all_fingers(cube_contacts())
        assert grasp_drift() < tolerance

    start_height = data.body("cube").xpos[2]
    robot.update_state()
    start_target = robot.target.position.copy()
    jacobian = robot.state.get_jacobian("grasp")[:3, :7]
    delta = np.linalg.pinv(jacobian) @ np.array([0, 0, 0.05])
    for fraction in np.linspace(0, 1, 2500):
        blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        robot.target = ControlTarget(start_target + blend * delta)
        sim.physics_step()
    mujoco.mj_forward(model, data)
    assert data.body("cube").xpos[2] > start_height + 0.04
    assert gripper.has_contact_on_all_fingers(cube_contacts())
    assert grasp_drift() < tolerance

    gripper.set_target(0.074)
    sim.run_steps(1000)
    mujoco.mj_forward(model, data)
    assert model.geom("floor").id in cube_contacts()
    assert not set().union(*gripper.finger_geom_ids) & cube_contacts()
    assert data.body("cube").xpos[2] < start_height - 0.05
    assert not data.warning.number.any()
