import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ASSET_PATH
from mujoco_lab.assets.loader import load_asset, load_robot_config

ASSETS = ASSET_PATH / "robot"


def rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


@pytest.mark.parametrize("name", ["panda"])
def test_forward_kinematics_matches_urdf(name):
    simulator = Simulator(load_asset(name))
    data = simulator.data
    urdf = ET.parse(ASSETS / name / "robot.urdf").getroot()
    joints = urdf.findall("joint")
    children = {j.find("child").attrib["link"] for j in joints}
    root = next(
        link.attrib["name"] for link in urdf.findall("link") if link.attrib["name"] not in children
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
    sim = Simulator(
        create_environment("warehouse"),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    model, data = sim.model, sim.data
    assert [model.actuator(i).name for i in range(model.nu)] == [
        *(f"panda/panda_joint{i}" for i in range(1, 8)),
        "panda/panda_finger_joint1",
    ]
    data.actuator("panda/panda_finger_joint1").ctrl[0] = 0.01
    mujoco.mj_step(model, data, nstep=1500)
    left = float(data.joint("panda/panda_finger_joint1").qpos[0])
    right = float(data.joint("panda/panda_finger_joint2").qpos[0])
    assert abs(left - right) < 1e-3
    assert left < 0.03
    assert int(data.warning.number.sum()) == 0
