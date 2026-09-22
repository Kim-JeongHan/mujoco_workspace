import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lab import ENVIRONMENT_NAMES, RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ASSET_PATH, ROBOT_NAMES, ROBOT_SCENES
from mujoco_lab.robot import _load_asset

ASSETS = ASSET_PATH / "robot"


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_bundled_robot_home_times_are_zero(name):
    spec, _ = _load_asset(name)
    assert spec.key("home").time == 0


def test_asset_home_is_composed_at_time_zero_with_one_compile(tmp_path, monkeypatch):
    path = tmp_path / "robot.xml"
    path.write_text("""<mujoco>
      <worldbody>
        <site name="world_site"/>
        <body name="root">
          <joint name="joint"/>
          <geom type="sphere" size="0.1"/>
          <site name="tip"/><site/>
        </body>
      </worldbody>
      <actuator><motor name="motor" joint="joint"/></actuator>
      <keyframe><key name="home" time="2" qpos="0.4" qvel="0.2" ctrl="0.3"/></keyframe>
    </mujoco>""")
    monkeypatch.setitem(ROBOT_SCENES, "test", path)
    compile_spec = mujoco.MjSpec.compile
    compilations = []

    def compile_once(spec):
        compilations.append(spec.modelname)
        return compile_spec(spec)

    monkeypatch.setattr(mujoco.MjSpec, "compile", compile_once)
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("arm", "test")])
    assert len(compilations) == 1
    robot = sim.robots["arm"]
    assert robot.state.site_id("tip") == sim.model.site("arm/tip").id
    with pytest.raises(ValueError, match="no site"):
        robot.state.site_id("world_site")
    sim.run_steps(3)
    sim.reset()
    assert sim.data.time == 0
    np.testing.assert_array_equal(sim.data.qpos, [0.4])
    np.testing.assert_array_equal(sim.data.qvel, [0.2])
    np.testing.assert_array_equal(sim.data.ctrl, [0.3])


@pytest.mark.parametrize("name,nv,nu", [("panda", 9, 8)])
@pytest.mark.parametrize("environment", ENVIRONMENT_NAMES)
def test_robot_home_pose_and_position_servos(name, nv, nu, environment):
    sim = Simulator(create_environment(environment), robots=[RobotSpec(name, name)])
    model, data = sim.model, sim.data
    robot = sim.robots[name]
    assert robot.state.nv == nv
    assert robot.nu == nu
    home = model.key(name + "/home")
    np.testing.assert_allclose(
        data.qpos[robot.state.qpos_indices], home.qpos[robot.state.qpos_indices]
    )
    np.testing.assert_allclose(data.ctrl[robot.actuator_ids], home.ctrl[robot.actuator_ids])
    assert all(
        not model.geom(contact.geom1).name.startswith(name + "/")
        and not model.geom(contact.geom2).name.startswith(name + "/")
        for contact in data.contact
    )
    mujoco.mj_step(model, data, nstep=1000)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert int(data.warning.number.sum()) == 0
    assert (
        np.max(np.abs(data.qpos[robot.state.qpos_indices] - home.qpos[robot.state.qpos_indices]))
        < 0.15
    )


@pytest.mark.parametrize("name", ["panda"])
def test_mjcf_joint_limits_and_link_masses_match_urdf(name):
    model = Simulator(mujoco.MjSpec.from_file(str(ASSETS / name / "scene.xml"))).model
    urdf = ET.parse(ASSETS / name / "robot.urdf").getroot()
    for link in urdf.findall("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            actual = float(model.body(link.attrib["name"]).mass[0])
            assert actual == pytest.approx(float(mass.attrib["value"]), rel=1e-5)
    for joint in urdf.findall("joint"):
        if joint.attrib["type"] in {"fixed", "continuous"}:
            continue
        limit = joint.find("limit")
        expected = [float(limit.attrib["lower"]), float(limit.attrib["upper"])]
        np.testing.assert_allclose(model.joint(joint.attrib["name"]).range, expected, atol=1e-5)


def rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


@pytest.mark.parametrize("name", ["panda"])
def test_forward_kinematics_matches_urdf(name):
    simulator = Simulator(mujoco.MjSpec.from_file(str(ASSETS / name / "scene.xml")))
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
    sim = Simulator(create_environment("warehouse"), robots=[RobotSpec("panda", "panda")])
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


@pytest.mark.parametrize("name", ["panda", "forte"])
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
