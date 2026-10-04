"""Verify named-asset inspection without assuming arm or gripper layout."""

import mujoco
import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("trimesh")

from mujoco_lab.assets import view_collision


@pytest.fixture(scope="module")
def previews():
    return {name: view_collision.RobotPreview(name) for name in ("panda", "forte", "forte2")}


@pytest.mark.parametrize("name", ["panda", "forte", "forte2"])
def test_named_assets_load_only_robot_and_keep_their_home(previews, name):
    preview = previews[name]
    model = preview.model
    np.testing.assert_allclose(preview.data.qpos, model.key("home").qpos, atol=1e-12)
    assert all(model.geom(i).name not in ("table", "floor") for i in range(model.ngeom))
    assert len(preview.controls) == 8
    assert [c.unit for c in preview.controls].count("mm") == 1
    if name == "panda":
        assert preview.values()["panda_joint4"] == pytest.approx(-2.35619449)
        assert preview.values()["panda_finger_joint1"] == pytest.approx(0.04)


@pytest.mark.parametrize("name", ["panda", "forte", "forte2"])
def test_gripper_coupling_and_home_reset_use_model_coordinates(previews, name):
    preview = previews[name]
    control = next(c for c in preview.controls if c.unit == "mm")
    value = (control.lower + control.upper) / 2
    preview.set_values({control.name: value})
    slide_joints = np.flatnonzero(preview.model.jnt_type == int(mujoco.mjtJoint.mjJNT_SLIDE))
    assert len(slide_joints) == 2
    actual = preview.data.qpos[preview.model.jnt_qposadr[slide_joints]]
    np.testing.assert_allclose(actual, value, atol=1e-12)
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


@pytest.mark.parametrize("name", ["forte", "forte2"])
def test_zero_and_home_preview_buttons_preserve_both_model_presets(previews, name):
    preview = previews[name]
    assert preview.zero is not None
    preview.reset_zero()
    np.testing.assert_array_equal(preview.data.qpos, 0)
    preview.reset_home()
    np.testing.assert_allclose(preview.data.qpos, preview.model.key("home").qpos, atol=1e-12)
    assert abs(preview.data.joint("shoulder_pitch").qpos[0]) > 1
