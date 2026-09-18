import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lab import ENVIRONMENT_NAMES, create_robot, load_simulation
from mujoco_lab.simulation import ASSET_ROOT as ASSETS


@pytest.mark.parametrize("name,nv,nu", [("ur20", 6, 6), ("ur30", 6, 6), ("panda", 9, 8)])
@pytest.mark.parametrize("environment", ENVIRONMENT_NAMES)
def test_robot_home_pose_and_position_servos(name, nv, nu, environment):
    model, data = create_robot(name, environment=environment)
    assert model.nv == nv
    assert model.nu == nu
    home = model.key("home")
    np.testing.assert_allclose(data.qpos, home.qpos)
    np.testing.assert_allclose(data.ctrl, home.ctrl)
    assert data.ncon == 0
    mujoco.mj_step(model, data, nstep=1000)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert int(data.warning.number.sum()) == 0
    assert np.max(np.abs(data.qpos - home.qpos)) < 0.15


@pytest.mark.parametrize("name", ["ur20", "ur30", "panda"])
def test_source_joint_limits_and_link_masses_are_preserved(name):
    model, _ = load_simulation(ASSETS / name / "scene.xml")
    source = ET.parse(ASSETS / name / "source/robot.urdf").getroot()
    for link in source.findall("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            actual = float(model.body(link.attrib["name"]).mass[0])
            assert actual == pytest.approx(float(mass.attrib["value"]), rel=1e-5)
    for joint in source.findall("joint"):
        if joint.attrib["type"] in {"fixed", "continuous"}:
            continue
        limit = joint.find("limit")
        expected = [float(limit.attrib["lower"]), float(limit.attrib["upper"])]
        np.testing.assert_allclose(model.joint(joint.attrib["name"]).range, expected, atol=1e-5)


def rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


@pytest.mark.parametrize("name", ["ur20", "ur30", "panda"])
def test_forward_kinematics_matches_original_urdf(name):
    model, data = load_simulation(ASSETS / name / "scene.xml")
    source = ET.parse(ASSETS / name / "source/robot.urdf").getroot()
    joints = source.findall("joint")
    children = {j.find("child").attrib["link"] for j in joints}
    root = next(
        link.attrib["name"]
        for link in source.findall("link")
        if link.attrib["name"] not in children
    )
    transforms = {root: np.eye(4)}
    pending = joints.copy()
    while pending:
        ready = [j for j in pending if j.find("parent").attrib["link"] in transforms]
        assert ready, "URDF joint graph contains an unresolved parent or cycle"
        for joint in ready:
            pending.remove(joint)
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            transform = np.eye(4)
            origin = joint.find("origin")
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                transform[:3, :3] = (
                    rotation([0, 0, 1], yaw)
                    @ rotation([0, 1, 0], pitch)
                    @ rotation([1, 0, 0], roll)
                )
            motion = np.eye(4)
            if joint.attrib["type"] != "fixed":
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                axis /= np.linalg.norm(axis)
                value = float(data.joint(joint.attrib["name"]).qpos[0])
                if joint.attrib["type"] == "prismatic":
                    motion[:3, 3] = axis * value
                else:
                    motion[:3, :3] = rotation(axis, value)
            transforms[child] = transforms[parent] @ transform @ motion
            np.testing.assert_allclose(data.body(child).xpos, transforms[child][:3, 3], atol=2e-5)
            np.testing.assert_allclose(
                data.body(child).xmat.reshape(3, 3), transforms[child][:3, :3], atol=2e-5
            )


def test_panda_fingers_remain_coupled_when_commanded():
    model, data = create_robot("panda", environment="warehouse")
    assert [model.actuator(i).name for i in range(model.nu)] == [
        *(f"panda_joint{i}" for i in range(1, 8)),
        "panda_finger_joint1",
    ]
    data.actuator("panda_finger_joint1").ctrl[0] = 0.01
    mujoco.mj_step(model, data, nstep=1500)
    left = float(data.joint("panda_finger_joint1").qpos[0])
    right = float(data.joint("panda_finger_joint2").qpos[0])
    assert abs(left - right) < 1e-3
    assert left < 0.03
    assert int(data.warning.number.sum()) == 0


@pytest.mark.parametrize("name", ["ur20", "ur30", "panda", "forte"])
def test_robot_files_resolve_inside_the_asset_directory(name):
    directory = ASSETS / name
    for filename in ["scene.xml", "robot.xml"]:
        for element in ET.parse(directory / filename).iter():
            reference = element.get("file")
            if reference:
                assert not Path(reference).is_absolute()
                target = (directory / reference).resolve()
                assert target.is_relative_to(directory.resolve())
                assert target.is_file()
