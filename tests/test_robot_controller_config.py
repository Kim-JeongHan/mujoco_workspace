"""Robot-local YAML defaults, joint gains, and task-specific tuning."""

import mujoco
import numpy as np
import pytest
import yaml
from dacite.exceptions import DaciteError

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import ControllerConfig, FortePDGain, GripperConfig, RobotConfig
from mujoco_lab.behaviors.book import create_book_controller
from mujoco_lab.control import JointSpacePD, PositionController, create_controller


@pytest.mark.parametrize(
    "robot_type,kind", [("forte", JointSpacePD), ("panda", PositionController)]
)
def test_default_controller_matches_robot_actuators(robot_type, kind):
    config = load_robot_config(robot_type)
    assert isinstance(config, RobotConfig)
    assert isinstance(config.controller, ControllerConfig)
    if robot_type == "forte":
        assert isinstance(config.controller.pd_gains, FortePDGain)
        assert config.controller.pd_gains.shoulder_yaw.kp == 147.0
    else:
        assert config.controller.pd_gains is None
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("arm", robot_type, config=config)],
    )
    robot = sim.robots["arm"]
    assert robot.config is not config
    assert robot.config.controller is config.controller
    assert robot.state.constraints is config.constraints
    robot.change_controller(create_controller(robot, config.controller))
    assert isinstance(robot.controller, kind)
    joint_names, slots = robot.get_arm_joint_mapping()
    assert config.constraints.joint_names == list(joint_names)
    assert len(config.constraints.velocity_limit) == len(joint_names)
    assert len(config.constraints.acceleration_limit) == len(joint_names)
    model_limits = robot.state.get_joint_limits(slots)
    configured_limits = np.asarray(config.constraints.position_limit)
    bounded = np.isfinite(model_limits).all(axis=1)
    np.testing.assert_allclose(configured_limits[bounded], model_limits[bounded], atol=1e-12)
    if not bounded.all():
        np.testing.assert_allclose(
            configured_limits[~bounded], np.tile([-2 * np.pi, 2 * np.pi], ((~bounded).sum(), 1))
        )
    assert isinstance(config.gripper, GripperConfig)
    assert robot.gripper.actuator_id == sim.model.actuator("arm/" + config.gripper.actuator).id
    assert config.gripper.velocity_limit == 0.2
    assert config.gripper.acceleration_limit == 1.0
    if robot_type == "forte":
        assert robot.controller.frame == config.controller.frame == "grasp"
    sim.step()
    assert np.isfinite(sim.data.qpos).all()
    sim.reset()
    assert robot.config is not config
    assert robot.config.controller is config.controller
    assert robot.state.constraints is config.constraints


def test_yaml_gains_and_book_tuning_are_independent(tmp_path, monkeypatch):
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("arm", "forte", config=load_robot_config("forte"))],
    )
    robot = sim.robots["arm"]
    names = robot.get_arm_joint_mapping()[0]
    gains = {name: {"kp": 100 + i, "kd": 10 + i} for i, name in enumerate(names)}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"controller": {"name": "pd", "pd_gains": gains}}))
    monkeypatch.setitem(ROBOT_ASSETS, "forte", RobotAsset(tmp_path / "robot.xml"))
    default = create_controller(robot, load_robot_config(robot.robot_type).controller)
    np.testing.assert_array_equal(default.kp, np.arange(100, 107))
    np.testing.assert_array_equal(default.kd, np.arange(10, 17))
    path.write_text(
        yaml.safe_dump(
            {
                "controller": {
                    "name": "pd",
                    "pd_gains": gains,
                    "gravity_compensation": False,
                },
            }
        )
    )
    custom = create_controller(robot, load_robot_config(robot.robot_type).controller)
    assert default.gravity_compensation
    assert not custom.gravity_compensation
    np.testing.assert_array_equal(custom.kp, default.kp)
    np.testing.assert_array_equal(custom.kd, default.kd)
    book = create_book_controller(robot, load_robot_config(robot.robot_type).controller)
    np.testing.assert_array_equal(book.kp[:4], default.kp[:4] * 4)
    np.testing.assert_array_equal(book.kp[4:], default.kp[4:] * 24)
    np.testing.assert_array_equal(book.kd[:4], default.kd[:4] * 2)
    np.testing.assert_allclose(book.kd[4:], default.kd[4:] * 2 * np.sqrt(6))
    np.testing.assert_array_equal(default.kp, np.arange(100, 107))
    path.write_text(yaml.safe_dump({"controller": {"name": "position"}}))
    with pytest.raises(ValueError, match="requires position"):
        create_controller(robot, load_robot_config(robot.robot_type).controller)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"controller": "pd"},
        {"controller": {"name": "invalid"}},
        {"controller": {"name": "pd", "pd_gains": [1, 2]}},
        {"controller": {"name": "pd", "pd_gains": {"joint": {"kp": 1}}}},
    ],
)
def test_yaml_schema_errors_propagate_from_classmethod(tmp_path, config):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(DaciteError):
        RobotConfig.load(path)


def test_update_replaces_asset_names_without_changing_settings():
    config = load_robot_config("forte")
    controller, constraints, gripper = config.controller, config.constraints, config.gripper
    first = mujoco.MjSpec.from_string("""<mujoco>
      <worldbody>
        <site name="world_site"/>
        <body name="root">
          <joint name="joint"/><geom size=".1"/>
          <site name="tip"/><site/>
        </body>
      </worldbody>
      <actuator><motor name="drive" joint="joint"/></actuator>
    </mujoco>""")
    config.update_model_info(first)
    assert config.model_info.joint_names == ("joint",)
    assert config.model_info.actuator_names == ("drive",)
    assert config.model_info.site_names == ("tip",)
    assert config.model_info.root_name == "root"
    previous = config.model_info
    second = mujoco.MjSpec.from_string("""<mujoco>
      <worldbody><body name="other"><geom size=".1"/><site name="other_tip"/></body></worldbody>
    </mujoco>""")
    config.update_model_info(second)
    assert config.model_info.joint_names == config.model_info.actuator_names == ()
    assert config.model_info.site_names == ("other_tip",)
    assert config.model_info.root_name == "other"
    assert previous.root_name == "root"
    assert config.controller is controller
    assert config.constraints is constraints
    assert config.gripper is gripper


def test_shared_config_gets_independent_asset_metadata(tmp_path, monkeypatch):
    from mujoco_lab.utils import Transform

    for name in ("a", "b"):
        path = tmp_path / f"{name}.xml"
        path.write_text(f"""<mujoco>
          <worldbody><body name="root_{name}">
            <joint name="joint_{name}"/><geom size=".1"/><site name="tip_{name}"/>
          </body></worldbody>
        </mujoco>""")
        monkeypatch.setitem(ROBOT_ASSETS, name, RobotAsset(path))
    config = RobotConfig(ControllerConfig(name="none"))
    initial_info = config.model_info
    sim = Simulator(
        create_environment("empty"),
        robots=[
            RobotSpec("left", "a", Transform(translation=[-0.5, 0, 1]), config=config),
            RobotSpec("right", "b", Transform(translation=[0.5, 0, 1]), config=config),
        ],
    )
    for name, suffix in (("left", "a"), ("right", "b")):
        robot = sim.robots[name]
        assert robot.config.model_info.root_name == f"root_{suffix}"
        assert robot.config.model_info.joint_names == (f"joint_{suffix}",)
        assert robot.state.site_id(f"tip_{suffix}") == sim.model.site(f"{name}/tip_{suffix}").id
    assert sim.robots["left"].config.model_info is not sim.robots["right"].config.model_info
    assert config.model_info is initial_info
