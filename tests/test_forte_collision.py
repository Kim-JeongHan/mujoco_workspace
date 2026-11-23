"""Forte proxy exclusions preserve a collision-free home and environment contacts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

ASSET = Path(__file__).resolve().parents[1] / "src/mujoco_lab/assets/robot/forte/robot.xml"


def pose_state(model, pose="zero"):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(pose).id)
    mujoco.mj_forward(model, data)
    return data


@pytest.mark.parametrize("pose", ["home", "zero"])
def test_forte_presets_have_no_proxy_self_contacts(pose):
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    assert pose_state(model, pose).ncon == 0


@pytest.mark.parametrize("angle", [-0.1, 0.1])
def test_forte_yaw_rotates_the_arm_about_positive_z(angle):
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = pose_state(model)
    joint = model.joint("shoulder_yaw").id
    origin = data.xanchor[joint, :2].copy()
    before = data.site("grasp").xpos[:2] - origin
    data.qpos[model.jnt_qposadr[joint]] = angle
    mujoco.mj_forward(model, data)
    after = data.site("grasp").xpos[:2] - origin
    rotation = np.arctan2(before[0] * after[1] - before[1] * after[0], before @ after)
    assert rotation == pytest.approx(angle, abs=1e-12)


@pytest.mark.parametrize("angle", [-np.pi, -np.pi / 2, 0])
def test_shoulder_pitch_rotates_from_its_cad_frame(angle):
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = pose_state(model)
    data.joint("shoulder_yaw").qpos[0] = 0.1
    mujoco.mj_forward(model, data)
    parent = data.body("shoulder_yaw_link").xmat.reshape(3, 3).copy()
    neutral = parent.T @ data.body("shoulder_pitch_link").xmat.reshape(3, 3)
    data.joint("shoulder_pitch").qpos[0] = angle
    mujoco.mj_forward(model, data)
    pitch = data.body("shoulder_pitch_link").xmat.reshape(3, 3)
    cosine, sine = np.cos(angle), np.sin(angle)
    expected = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    np.testing.assert_allclose(parent.T @ pitch, expected @ neutral, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "name,angle", [("shoulder_pitch", -0.1), ("elbow_pitch", 0.1), ("wrist_pitch", 0.1)]
)
def test_pitch_joints_share_positive_y_rotation(name, angle):
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = pose_state(model)
    joint = model.joint(name)
    body = int(joint.bodyid[0])
    before = data.xmat[body].reshape(3, 3).copy()
    data.joint(name).qpos[0] = angle
    mujoco.mj_forward(model, data)
    delta = data.xmat[body].reshape(3, 3) @ before.T
    # The Y component of the rotation axis times sin(angle), in world coordinates.
    positive_y = (delta[0, 2] - delta[2, 0]) / 2
    assert positive_y == pytest.approx(np.sin(angle), abs=1e-10)


@pytest.mark.parametrize("roll", [-0.7, 0, 0.8])
def test_forearm_and_wrist_axes_align_with_elbow_at_zero(roll):
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = pose_state(model)
    data.joint("shoulder_roll").qpos[0] = roll
    mujoco.mj_forward(model, data)
    elbow = data.body("elbow_link").xmat.reshape(3, 3)
    joints = [model.joint(name).id for name in ("lower_arm_roll", "wrist_pitch", "wrist_yaw")]
    np.testing.assert_allclose(data.xaxis[joints] @ elbow, np.eye(3), rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "geom_name", ["upper_arm_capsule", "shoulder_roll_capsule", "shoulder_hinge_capsule"]
)
def test_forte_exclusions_keep_contacts_with_external_geometry(geom_name):
    spec = mujoco.MjSpec.from_file(str(ASSET))
    model = spec.compile()
    center = pose_state(model).geom(geom_name).xpos.copy()
    spec.worldbody.add_geom(
        name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005], pos=center
    )
    model = spec.compile()
    data = pose_state(model)
    probe = model.geom("probe").id
    collision = model.geom(geom_name).id
    assert any({contact.geom1, contact.geom2} == {probe, collision} for contact in data.contact)


def test_shoulder_proxies_stay_with_the_hinge_when_upper_arm_rolls():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = pose_state(model)
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
    data = pose_state(model)
    body = model.body("elbow_link").id
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
    for angle in np.linspace(*model.jnt_range[joint], 37):
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
            assert np.maximum.reduce(clearances).min() >= -1e-9
