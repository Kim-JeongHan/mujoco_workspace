import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from mujoco_lab import ENVIRONMENT_NAMES, create_robot
from mujoco_lab.simulation import ASSET_ROOT, initialize_data

ASSET = ASSET_ROOT / "forte"
JOINTS = (
    "shoulder_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_pitch",
    "lower_arm_roll",
    "wrist_pitch",
    "wrist_roll",
)


def source_model():
    original = ET.parse(ASSET / "source/forte.xml").getroot()
    original.find("compiler").set("meshdir", str(ASSET / "meshes"))
    return mujoco.MjModel.from_xml_string(ET.tostring(original, encoding="unicode"))


def test_forte_keeps_upstream_torque_motor_interface():
    model, data = create_robot("forte")
    assert (model.nq, model.nv, model.nu) == (7, 7, 7)
    assert [model.joint(i).name for i in range(7)] == list(JOINTS)
    assert [model.actuator(i).name for i in range(7)] == [name + "_motor" for name in JOINTS]
    limits = np.array([87, 87, 87, 87, 87, 12, 12])
    np.testing.assert_allclose(model.actuator_ctrlrange, np.column_stack([-limits, limits]))
    np.testing.assert_allclose(model.actuator_gear[:, 0], 1)
    np.testing.assert_allclose(data.ctrl, 0)
    np.testing.assert_allclose(data.qpos, model.key("home").qpos)
    assert np.all(model.dof_armature > 0)
    assert model.site("ee_site").id >= 0


def test_forte_preserves_original_kinematics_and_dynamics():
    original = source_model()
    original_data = initialize_data(original)
    model, data = create_robot("forte")
    for index in range(1, original.nbody):
        name = original.body(index).name
        for attribute in ["mass", "inertia", "ipos", "iquat"]:
            np.testing.assert_allclose(
                getattr(model.body(name), attribute),
                getattr(original.body(name), attribute),
                atol=1e-12,
            )
        np.testing.assert_allclose(data.body(name).xpos, original_data.body(name).xpos, atol=1e-12)
        np.testing.assert_allclose(data.body(name).xmat, original_data.body(name).xmat, atol=1e-12)
    for attribute in ["jnt_axis", "jnt_range", "dof_armature", "dof_damping"]:
        np.testing.assert_allclose(getattr(model, attribute), getattr(original, attribute))


def test_forte_preserves_the_four_upstream_collision_shapes():
    original = source_model()
    model, _ = create_robot("forte")
    names = [
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom(i).name != "floor" and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    assert set(names) == {
        "shoulder_yaw_link_collision_0",
        "upper_arm_link_collision_0",
        "lower_arm_link_collision_0",
        "wrist_link_collision_0",
    }
    for name in names:
        for attribute in ["type", "pos", "quat", "size", "friction", "contype", "conaffinity"]:
            np.testing.assert_allclose(
                getattr(model.geom(name), attribute),
                getattr(original.geom(name), attribute),
                atol=1e-12,
            )


@pytest.mark.parametrize("environment", ENVIRONMENT_NAMES)
def test_forte_zero_torque_simulation_is_finite(environment):
    model, data = create_robot("forte", environment=environment)
    initial = data.qpos.copy()
    assert data.ncon == 0
    mujoco.mj_step(model, data, nstep=1000)
    assert data.time == pytest.approx(2)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert int(data.warning.number.sum()) == 0
    # The source uses torque motors with zero input, rather than pose-holding servos.
    assert np.max(np.abs(data.qpos - initial)) > 0.1


def test_forte_accepts_gravity_compensation_torques():
    model, data = create_robot("forte")
    initial = data.qpos.copy()
    for _ in range(500):
        data.ctrl[:] = data.qfrc_bias
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos, initial, atol=1e-6)
    assert int(data.warning.number.sum()) == 0


def test_forte_source_and_mesh_bytes_match_recorded_hashes():
    manifest = json.loads((ASSET / "SOURCE.json").read_text())
    for original, expected in manifest["source_files"].items():
        path = (
            ASSET / "source/forte.xml"
            if original.endswith(".xml")
            else (ASSET / "meshes" / original.rsplit("/", 1)[1])
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
