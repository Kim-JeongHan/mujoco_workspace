"""Verify named-asset inspection without assuming arm or gripper layout."""

import mujoco
import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("trimesh")

from mujoco_lab.assets import view_collision


@pytest.fixture(scope="module")
def previews():
    return {name: view_collision.RobotPreview(name) for name in ("panda", "forte", "forte_backup")}


@pytest.mark.parametrize("name", ["panda", "forte", "forte_backup"])
def test_gripper_coupling_and_home_reset_use_model_coordinates(previews, name):
    preview = previews[name]
    control = next(c for c in preview.controls if c.unit == "mm")
    value = (control.lower + control.upper) / 2
    preview.set_values({control.name: value})
    slide_joints = np.flatnonzero(preview.model.jnt_type == int(mujoco.mjtJoint.mjJNT_SLIDE))
    assert len(slide_joints) == 2
    actual = preview.data.qpos[preview.model.jnt_qposadr[slide_joints]]
    np.testing.assert_allclose(actual, value / 2, atol=1e-12)
    assert preview.values()[control.name] == pytest.approx(value)
    preview.reset_home()
    np.testing.assert_allclose(preview.data.qpos, preview.home, atol=1e-12)


def test_new_directory_and_nonstandard_joint_layout_need_no_registration(tmp_path, monkeypatch):
    directory = tmp_path / "custom_arm"
    directory.mkdir()
    (directory / "robot.xml").write_text("""<mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="floating"><freejoint/><geom type="sphere" size="0.03"/></body>
        <body name="driver_body">
          <joint name="driver" type="slide" ref="0.01" range="0 0.1"/>
          <geom type="sphere" size="0.01"/>
          <body name="follower_body">
            <joint name="follower" type="slide" ref="0.02" range="0 0.3"/>
            <geom type="sphere" size="0.01"/>
            <body name="leaf_body">
              <joint name="leaf" type="slide" ref="-0.01" range="-0.02 0.2"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </body>
        </body>
        <body name="hinge_body">
          <joint name="hinge" ref="0.2" range="-1 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
      </worldbody>
      <equality>
        <joint joint1="leaf" joint2="follower" polycoef="0 0.5 0 0 0"/>
        <joint joint1="follower" joint2="driver" polycoef="0.005 2 0.25 0 0"/>
      </equality>
      <keyframe><key name="home"
        qpos="0.1 0.2 0.3 1 0 0 0 0.035 0.07515625 0.017578125 0.4"/></keyframe>
    </mujoco>""")
    monkeypatch.setattr(view_collision, "ROBOT_PATH", tmp_path)
    assert view_collision.robot_names() == ("custom_arm",)
    preview = view_collision.RobotPreview("custom_arm")
    assert [c.name for c in preview.controls] == ["driver", "hinge"]
    np.testing.assert_allclose(preview.data.qpos, preview.home, atol=1e-12)
    preview.set_values({"driver": 0.06, "hinge": -0.3})
    assert preview.data.joint("follower").qpos[0] == pytest.approx(0.125625)
    assert preview.data.joint("leaf").qpos[0] == pytest.approx(0.0428125)
    np.testing.assert_array_equal(preview.data.qpos[:7], preview.home[:7])
    assert preview.data.joint("hinge").qpos[0] == pytest.approx(-0.3)
    preview.reset_home()
    np.testing.assert_allclose(preview.data.qpos, preview.home, atol=1e-12)
    with pytest.raises(ValueError, match="Unknown robot"):
        view_collision.RobotPreview("missing")
