"""Forte proxy exclusions preserve a collision-free home and environment contacts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

ASSET = Path(__file__).resolve().parents[1] / "src/mujoco_lab/assets/robot/forte/robot.xml"


def home_state(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def test_forte_home_has_no_proxy_self_contacts():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    assert home_state(model).ncon == 0


@pytest.mark.parametrize(
    "geom_name", ["upper_arm_capsule", "shoulder_roll_capsule", "shoulder_hinge_capsule"]
)
def test_forte_exclusions_keep_contacts_with_external_geometry(geom_name):
    spec = mujoco.MjSpec.from_file(str(ASSET))
    model = spec.compile()
    center = home_state(model).geom(geom_name).xpos.copy()
    spec.worldbody.add_geom(
        name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005], pos=center
    )
    model = spec.compile()
    data = home_state(model)
    probe = model.geom("probe").id
    collision = model.geom(geom_name).id
    assert any({contact.geom1, contact.geom2} == {probe, collision} for contact in data.contact)


def test_shoulder_proxies_stay_with_the_hinge_when_upper_arm_rolls():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = home_state(model)
    geoms = [model.geom(name).id for name in ("shoulder_roll_capsule", "shoulder_hinge_capsule")]
    positions = data.geom_xpos[geoms].copy()
    rotations = data.geom_xmat[geoms].copy()
    joint = model.joint("shoulder_roll").id
    for angle in np.linspace(model.jnt_range[joint, 0], model.jnt_range[joint, 1], 7):
        data.qpos[model.jnt_qposadr[joint]] = angle
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(data.geom_xpos[geoms], positions, atol=1e-12)
        np.testing.assert_allclose(data.geom_xmat[geoms], rotations, atol=1e-12)


def test_arm_and_remaining_elbow_capsules_cover_cad_through_pitch_rotation():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = home_state(model)
    body = model.body("elbowlink").id
    meshes = []
    for geom in np.flatnonzero(
        (model.geom_bodyid == body) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    ):
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        meshes.append((geom, model.mesh_vert[start : start + count]))
    proxies = [
        model.geom(name).id
        for name in ("upper_arm_capsule", "elbow_upper_capsule", "elbow_link_capsule")
    ]
    joint = model.joint("elbow_pitch").id
    for angle in np.linspace(-np.pi, np.pi, 37):
        data.qpos[model.jnt_qposadr[joint]] = angle
        mujoco.mj_kinematics(model, data)
        for geom, vertices in meshes:
            points = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            clearances = []
            for proxy in proxies:
                local = (points - data.geom_xpos[proxy]) @ data.geom_xmat[proxy].reshape(3, 3)
                radius, half_length = model.geom_size[proxy, :2]
                axial = np.maximum(np.abs(local[:, 2]) - half_length, 0)
                distances = np.sqrt(np.sum(local[:, :2] ** 2, axis=1) + axial**2)
                clearances.append(radius - distances)
            assert np.maximum.reduce(clearances).min() >= 0.001 - 1e-9
