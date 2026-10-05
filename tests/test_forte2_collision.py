"""Forte2 proxy exclusions preserve a collision-free home and environment contacts."""

from pathlib import Path

import mujoco

ASSET = Path(__file__).resolve().parents[1] / "src/mujoco_lab/assets/robot/forte2/robot.xml"


def home_state(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def test_forte2_home_has_no_proxy_self_contacts():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    assert home_state(model).ncon == 0


def test_forte2_exclusions_keep_contacts_with_external_geometry():
    spec = mujoco.MjSpec.from_file(str(ASSET))
    model = spec.compile()
    center = home_state(model).geom("upper_arm_capsule").xpos.copy()
    spec.worldbody.add_geom(
        name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005], pos=center
    )
    model = spec.compile()
    data = home_state(model)
    probe = model.geom("probe").id
    arm = model.geom("upper_arm_capsule").id
    assert any({contact.geom1, contact.geom2} == {probe, arm} for contact in data.contact)
