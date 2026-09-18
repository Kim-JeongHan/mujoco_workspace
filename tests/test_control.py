import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lab import ENVIRONMENT_NAMES, create_environment, create_robot
from mujoco_lab.control import create_controller, run_steps
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.runner import apply_control
from mujoco_lab.control.visualization import annotate_controller
from mujoco_lab.robot import ROBOT_SCENES
from mujoco_lab.simulation import initialize_data
from mujoco_lab.state import read_state
from mujoco_lab.state.dynamics import Dynamics
from mujoco_lab.viewer import scoped_control_callback

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("mode", ["pd", "osc"])
def test_controller_torques_match_original_source(mode, monkeypatch):
    scripts = ROOT / "third_party/forte-arm-isaac-mujoco-demos/Forte_mujoco/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("pd_control" if mode == "pd" else "osc_control")
    original_type = module.JointSpacePD if mode == "pd" else module.OperationalSpaceControl
    model, data = create_robot("forte")
    original = original_type(model)
    connected = create_controller(mode, model, data)
    rng = np.random.default_rng(42)
    home = data.qpos.copy()
    for elapsed in [0.0, 0.4, 1.5, 1.9, 2.0, 3.2, 7.0, 11.4, 12.0]:
        data.time = elapsed
        data.qpos[:] = home + rng.uniform(-0.15, 0.15, model.nq)
        data.qvel[:] = rng.uniform(-0.5, 0.5, model.nv)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            connected.torques(read_state(data)),
            original.torques(model, data),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("mode", ["pd", "osc"])
def test_full_simulation_trace_matches_original_source(mode, monkeypatch):
    scripts = ROOT / "third_party/forte-arm-isaac-mujoco-demos/Forte_mujoco/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    source = importlib.import_module("forte_control")
    module = importlib.import_module("pd_control" if mode == "pd" else "osc_control")
    original_type = module.JointSpacePD if mode == "pd" else module.OperationalSpaceControl
    model, data = create_robot("forte")
    original_data = initialize_data(model)
    original = original_type(model)
    connected = create_controller(mode, model, data)
    original_stats = source.RunStats()
    traces = np.empty((2, 6000, model.nq + model.nv + model.nu))
    for step in range(6000):
        source._apply(model, original_data, original, original_stats)
        mujoco.mj_step(model, original_data)
        apply_control(model, data, connected)
        mujoco.mj_step(model, data)
        for index, state in enumerate((original_data, data)):
            traces[index, step] = np.concatenate([state.qpos, state.qvel, state.ctrl])
    np.testing.assert_allclose(traces[1], traces[0], rtol=1e-11, atol=1e-11)
    assert data.time == original_data.time
    np.testing.assert_array_equal(data.warning.number, original_data.warning.number)


def test_clipping_and_statistics_match_original_source(monkeypatch):
    scripts = ROOT / "third_party/forte-arm-isaac-mujoco-demos/Forte_mujoco/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    source = importlib.import_module("forte_control")
    pd = importlib.import_module("pd_control")
    model, data = create_robot("forte")
    original = pd.JointSpacePD(model)
    connected = create_controller("pd", model, data)
    original_stats = source.RunStats()
    from mujoco_lab.control.stats import RunStats

    stats = RunStats()
    data.qpos[:] += 0.8
    data.qvel[:] = 25.0
    mujoco.mj_forward(model, data)
    source._apply(model, data, original, original_stats)
    expected = data.ctrl.copy()
    apply_control(model, data, connected, stats)
    np.testing.assert_array_equal(data.ctrl, expected)
    assert stats.saturated_steps == original_stats.saturated_steps == 1
    assert stats.errors == original_stats.errors
    assert stats.describe("error", "rad") == original_stats.describe("error", "rad")


def test_osc_accepts_dynamics_supplied_as_numpy_snapshots(monkeypatch):
    scripts = ROOT / "third_party/forte-arm-isaac-mujoco-demos/Forte_mujoco/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    source = importlib.import_module("osc_control")
    model, data = create_robot("forte")
    data.qvel[:] = np.linspace(-0.2, 0.2, model.nv)
    data.time = 3.0
    mujoco.mj_forward(model, data)
    native = Dynamics(model, data)
    jacobian = native.get_jacobian("ee_site").copy()
    mass = native.get_mass_matrix().copy()
    position = native.get_frame_position("ee_site").copy()

    class SnapshotDynamics:
        nv = model.nv

        def get_frame_position(self, frame):
            return position

        def get_jacobian(self, frame):
            return jacobian

        def get_mass_matrix(self):
            return mass

    controller = OperationalSpaceControl(SnapshotDynamics())
    np.testing.assert_array_equal(
        controller.torques(read_state(data)),
        source.OperationalSpaceControl(model).torques(model, data),
    )


def test_controller_math_imports_and_runs_without_simulator_modules(tmp_path):
    code = """
import importlib.abc
import sys

class NoSimulatorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'mujoco_warp', 'isaaclab', 'isaacsim'}:
            raise ImportError(fullname)

sys.meta_path.insert(0, NoSimulatorImports())
import numpy as np
from mujoco_lab.control import JointSpacePD, OperationalSpaceControl
from mujoco_lab.control.trajectory import HOME_QPOS
from mujoco_lab.state import JointState
state = JointState(0.0, HOME_QPOS.copy(), np.zeros(7), np.ones(7))
np.testing.assert_array_equal(JointSpacePD().torques(state), np.ones(7))

class Dynamics:
    nv = 7
    def get_frame_position(self, frame):
        return np.array([0.65, 0.0, 0.53])
    def get_jacobian(self, frame):
        return np.eye(6, 7)
    def get_mass_matrix(self):
        return np.eye(7)

np.testing.assert_array_equal(OperationalSpaceControl(Dynamics()).torques(state), np.ones(7))
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


@pytest.mark.parametrize("mode,limit", [("pd", 0.05), ("osc", 0.01)])
def test_connected_controllers_track_full_reference_routines(mode, limit):
    model, data = create_robot("forte")
    controller = create_controller(mode, model, data)
    stats = run_steps(model, data, 6000, controller)
    assert stats.steps == 6000
    assert stats.saturated_steps == 0
    assert max(stats.errors) < limit
    assert int(data.warning.number.sum()) == 0


@pytest.mark.parametrize("mode", ["pd", "osc"])
@pytest.mark.parametrize("environment", ENVIRONMENT_NAMES)
def test_feedback_control_runs_in_each_environment(mode, environment):
    model, data = create_robot("forte", environment=environment)
    stats = run_steps(model, data, 500, create_controller(mode, model, data))
    assert stats.steps == 500
    assert np.isfinite(data.qpos).all()
    assert int(data.warning.number.sum()) == 0
    assert max(stats.errors) < 0.05


def test_osc_trajectory_and_annotations_follow_translated_rotated_mount():
    model, data = create_robot("forte")
    original = create_controller("osc", model, data)
    spec = create_environment("empty")
    mount = spec.site("robot_mount")
    mount.pos = [0.1, -0.2, 0.8]
    mount.quat = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    spec.attach(mujoco.MjSpec.from_file(str(ROBOT_SCENES["forte"])), prefix="", site=mount)
    moved_model = spec.compile()
    moved_data = initialize_data(moved_model)
    moved = create_controller("osc", moved_model, moved_data)
    rotation = moved_data.body("base_link").xmat.reshape(3, 3)
    translation = moved_data.body("base_link").xpos
    for elapsed in [0.0, 1.25, 2.5]:
        reference = original.circle(elapsed)
        actual = moved.circle(elapsed)
        np.testing.assert_allclose(actual[0], translation + rotation @ reference[0], atol=1e-12)
        np.testing.assert_allclose(actual[1], rotation @ reference[1], atol=1e-12)
        np.testing.assert_allclose(actual[2], rotation @ reference[2], atol=1e-12)
    apply_control(moved_model, moved_data, moved)
    scene = mujoco.MjvScene(moved_model, maxgeom=100)
    scene.ngeom = 0
    annotate_controller(scene, moved_model, moved_data, moved)
    np.testing.assert_allclose(scene.geoms[0].pos, moved.circle(0)[0], atol=1e-6)


def test_gui_callback_controls_only_its_model_and_restores_host_callback():
    model, data = create_robot("forte")
    other, other_data = create_robot("panda")
    controller = create_controller("pd", model, data)
    prior = mujoco.get_mjcb_control()
    host_calls = []

    def host(active_model, active_data):
        host_calls.append(active_model)

    mujoco.set_mjcb_control(host)
    try:
        with pytest.raises(RuntimeError, match="test exit"):
            with scoped_control_callback(model, data, controller):
                mujoco.mj_step(model, data)
                assert np.max(np.abs(data.ctrl)) > 0.1
                assert not host_calls
                mujoco.mj_step(other, other_data)
                assert other in host_calls
                raise RuntimeError("test exit")
        assert mujoco.get_mjcb_control() is host
    finally:
        mujoco.set_mjcb_control(prior)


@pytest.mark.parametrize("robot", ["ur20", "ur30", "panda"])
def test_forte_control_rejects_other_robot_actuators(robot):
    model, data = create_robot(robot)
    with pytest.raises(ValueError, match="requires the Forte"):
        create_controller("pd", model, data)


@pytest.mark.parametrize("mode", ["pd", "osc"])
def test_cli_connects_controller_to_workspace_environment(mode, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "simulate",
            "--robot",
            "forte",
            "--environment",
            "warehouse",
            "--controller",
            mode,
            "--steps",
            "100",
        ],
        cwd=tmp_path,
        env={**os.environ, "MUJOCO_GL": "disable"},
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["controller"] == mode
    assert output["warnings"] == 0
    assert output["tracking_error_max"] is not None


def test_cli_rejects_pd_on_position_controlled_robot(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mujoco_lab", "simulate", "--robot", "panda", "--controller", "pd"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires the Forte" in result.stderr
