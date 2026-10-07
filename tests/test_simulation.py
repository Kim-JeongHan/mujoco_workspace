import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config


def test_scene_blueprint_can_be_reused_without_accumulating_robot_attachments():
    scene = create_environment("empty")
    first = Simulator(scene, robots=[RobotSpec("arm", "forte", config=load_robot_config("forte"))])
    second = Simulator(scene, robots=[RobotSpec("arm", "forte", config=load_robot_config("forte"))])

    assert scene.body("arm/base_link") is None
    assert first.model.nbody == second.model.nbody
    np.testing.assert_array_equal(first.data.qpos, second.data.qpos)


def test_home_state_is_captured_after_forward_and_fully_restored():
    scene = mujoco.MjSpec.from_string("""<mujoco>
      <worldbody>
        <body name="jointed"><joint name="hinge"/><geom size="0.1"/></body>
        <body name="marker" mocap="true"><geom size="0.01"/></body>
      </worldbody>
      <actuator><general joint="hinge" dyntype="filter" dynprm="0.1"/></actuator>
      <keyframe><key name="home" time="2" qpos="0.4" qvel="0.2" act="0.3"
        ctrl="0.5" mpos="1 2 3" mquat="1 0 0 0"/></keyframe>
    </mujoco>""")
    sim = Simulator(scene)
    expected = {
        name: getattr(sim.data, name).copy()
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "xpos")
    }
    assert sim.data.time == 2
    sim.data.time = 9
    for name in expected:
        getattr(sim.data, name)[:] = 0

    sim.reset()

    assert sim.data.time == 2
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(sim.data, name), value)


def test_reset_failure_releases_lifecycle_and_allows_retry(monkeypatch):
    sim = Simulator(mujoco.MjSpec.from_string("<mujoco/>"))
    failure = RuntimeError("reset failed")

    def fail(*args):
        assert sim._state.get_state() == "resetting"
        sim._stop_requested = True
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(mujoco, "mj_setState", fail)
        with pytest.raises(RuntimeError, match="reset failed") as raised:
            sim.reset()
    assert raised.value is failure
    assert sim._state.get_state() == "idle"
    assert not sim._stop_requested
    sim.reset()
    sim.run_steps(1)
    assert sim._state.get_state() == "idle"


@pytest.mark.parametrize("field", ["qpos", "qvel"])
def test_divergence_releases_lifecycle_and_allows_reset(field):
    scene = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body><joint/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    sim = Simulator(scene)
    getattr(sim.data, field)[:] = np.nan
    with pytest.raises(RuntimeError, match="Simulation diverged"):
        sim.run_steps(0)
    assert sim._state.get_state() == "idle"
    assert not sim._stop_requested
    sim.reset()
    assert np.isfinite(getattr(sim.data, field)).all()
    sim.run_steps(1)
    assert sim._state.get_state() == "idle"
