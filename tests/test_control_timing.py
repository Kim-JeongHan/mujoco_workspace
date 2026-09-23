import os
import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_environment
from mujoco_lab.cli import Config
from mujoco_lab.control import Controller
from mujoco_lab.utils import Transform


class RecordingController(Controller):
    def __init__(self, value=None):
        self.value = value
        self.times = []

    def compute(self, state, target):
        self.times.append(state.time)
        self.tracking_error = float(len(self.times))
        return np.full(state.qpos.shape, len(self.times) if self.value is None else self.value)

    def reset(self):
        self.times.clear()
        self.tracking_error = 0.0


def make_simulator():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("arm", "forte")],
        dt=0.001,
    )
    controller = RecordingController()
    sim.robots["arm"].change_controller(controller)
    return sim, controller


def manager_with(simulator, name="simulator"):
    manager = SimulatorManager()
    manager.add_simulator(name, simulator)
    return manager


def test_fixed_default_period_and_explicit_override_leave_scene_unchanged():
    scene = create_environment("table_shelf")
    scene.option.timestep = 0.003
    default = Simulator(scene)
    overridden = Simulator(scene, dt=0.001)
    assert Config(command="view").dt == 0.002
    assert scene.option.timestep == 0.003
    assert default.dt == default.model.opt.timestep == 0.002
    assert overridden.dt == overridden.model.opt.timestep == 0.001


class PassiveViewer:
    def __init__(self, steps, *, on_lock=None, on_sync=None):
        self.steps = steps
        self.on_lock = on_lock
        self.on_sync = on_sync
        self.sync_calls = 0
        self.closed = False
        self._sim = lambda: None

    def is_running(self):
        return self.sync_calls < self.steps

    def lock(self):
        if self.on_lock is not None:
            self.on_lock()
        return nullcontext()

    def sync(self, *, state_only=False):
        assert state_only
        self.sync_calls += 1
        if self.on_sync is not None:
            self.on_sync()

    def close(self):
        self.closed = True


def test_every_step_updates_control_and_chunks_preserve_results():
    sim, controller = make_simulator()
    reference, other = make_simulator()
    assert sim.model.opt.timestep == sim.dt == 0.001
    assert sim.run_steps(0)["arm"].steps == 0
    assert not controller.times
    controls, errors = [], []
    for _ in range(11):
        result = sim.step()["arm"]
        assert result.steps == 1
        errors.extend(result.errors)
        controls.append(sim.data.ctrl[0])
    reference.run_steps(2)
    reference.step()
    reference.run_steps(8)
    np.testing.assert_allclose(controller.times, np.arange(11) * 0.001, atol=1e-15)
    assert controller.times == other.times
    assert controls == list(range(1, 12))
    assert errors == list(range(1, 12))
    assert sim.data.time == pytest.approx(0.011)
    np.testing.assert_array_equal(sim.data.qpos, reference.data.qpos)
    np.testing.assert_array_equal(sim.data.ctrl, reference.data.ctrl)


def test_stop_ends_headless_run_after_current_tick_and_does_not_affect_next_run():
    sim, controller = make_simulator()
    sim.stop()  # An idle stop request has no effect on the next run.
    updates = []

    def update(current):
        updates.append(current.data.time)
        if len(updates) == 3:
            current.stop()

    sim.target_updater = update
    stats = sim.run_steps(100)["arm"]
    assert stats.steps == len(controller.times) == 3
    np.testing.assert_allclose(updates, [0, 0.001, 0.002], atol=1e-15)
    assert sim.data.time == pytest.approx(0.003)
    assert sim._state.get_state() == "idle"
    assert sim.run_steps(2)["arm"].steps == 2
    assert len(controller.times) == 5


def test_stop_closes_passive_viewer_after_current_tick():
    sim, _ = make_simulator()
    manager = manager_with(sim)
    viewer = PassiveViewer(100)

    def update(current):
        assert current._state.get_state() == "viewing"
        current.stop()

    sim.target_updater = update
    with patch("mujoco.viewer.launch_passive", return_value=viewer):
        manager.show("simulator")
    assert viewer.sync_calls == 1 and viewer.closed
    assert sim.data.time == pytest.approx(0.001)
    assert sim._state.get_state() == "idle"


def test_saturation_and_errors_are_counted_every_physics_step():
    sim, _ = make_simulator()
    controller = RecordingController(1000)
    sim.robots["arm"].change_controller(controller)
    result = sim.run_steps(7)["arm"]
    assert result.steps == result.saturated_steps == 7
    assert result.errors == list(range(1, 8))
    np.testing.assert_allclose(controller.times, np.arange(7) * 0.001)


def test_control_replaces_external_inputs_on_the_next_tick():
    sim, _ = make_simulator()
    robot = sim.robots["arm"]
    controller = RecordingController(1000)
    robot.change_controller(controller)
    assert sim.step()["arm"].saturated_steps == 1
    sim.data.ctrl[robot.actuator_ids] = 0.25
    result = sim.step()["arm"]
    assert result.saturated_steps == 1 and result.errors == [2]
    assert controller.times == [0, 0.001]
    assert not np.all(sim.data.ctrl[robot.actuator_ids] == 0.25)
    robot.update_state()
    assert robot.control()
    result = sim.step()["arm"]
    assert result.saturated_steps == 1 and result.errors == [4]
    sim.run_steps(2)
    np.testing.assert_allclose(controller.times, [0, 0.001, 0.002, 0.002, 0.003, 0.004])


def test_reset_and_controller_replacement_apply_on_next_tick():
    sim, controller = make_simulator()
    sim.step()
    replacement = RecordingController(3)
    sim.robots["arm"].change_controller(replacement)
    sim.step()
    np.testing.assert_allclose(replacement.times, [0.001])
    assert controller.times == [0]
    sim.reset()
    assert not replacement.times
    assert sim.dt == 0.001
    sim.step()
    assert replacement.times == [0]
    sim.robots["arm"].change_controller(None)
    inputs = sim.data.ctrl.copy()
    result = sim.run_steps(3)["arm"]
    assert result.errors == [] and result.saturated_steps == 0
    np.testing.assert_array_equal(sim.data.ctrl, inputs)


def test_multiple_robots_share_one_clock_and_uncontrolled_inputs_survive():
    sim = Simulator(
        create_environment("empty"),
        robots=[
            RobotSpec("left", "forte", Transform(translation=[-0.8, 0, 0])),
            RobotSpec("right", "panda", Transform(translation=[0.8, 0, 0])),
        ],
        dt=0.001,
    )
    controller = RecordingController()
    sim.robots["left"].change_controller(controller)
    panda = sim.robots["right"]
    targets = sim.data.ctrl[panda.actuator_ids].copy()
    targets[0] += 0.01
    sim.data.ctrl[panda.actuator_ids] = targets
    stats = sim.run_steps(10)
    np.testing.assert_allclose(controller.times, np.arange(10) * 0.001)
    assert all(result.steps == 10 for result in stats.values())
    assert stats["right"].errors == []
    assert sim.data.time == pytest.approx(0.01)
    np.testing.assert_array_equal(sim.data.ctrl[panda.actuator_ids], targets)


def test_passive_viewer_matches_headless_timing_and_rk4_trajectory_after_priming():
    shown, shown_controller = make_simulator()
    reference, reference_controller = make_simulator()
    shown_updates, reference_updates = [], []
    shown.target_updater = lambda sim: shown_updates.append(sim.data.time)
    reference.target_updater = lambda sim: reference_updates.append(sim.data.time)
    shown.model.opt.integrator = reference.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    shown.run_steps(3)
    reference.run_steps(3)
    viewer = PassiveViewer(9)

    def launch(model, data):
        # Match launch_passive's live-data forward call. Manager.show() must undo its
        # cache-phase mutation before the first shared physics tick.
        mujoco.mj_forward(model, data)
        return viewer

    manager = manager_with(shown)
    with patch("mujoco.viewer.launch_passive", side_effect=launch):
        manager.show("simulator")
    reference.run_steps(9)
    np.testing.assert_allclose(shown_controller.times, reference_controller.times, atol=1e-15)
    np.testing.assert_allclose(shown_updates, reference_updates, atol=1e-15)
    np.testing.assert_array_equal(shown.data.qpos, reference.data.qpos)
    np.testing.assert_array_equal(shown.data.qvel, reference.data.qvel)
    np.testing.assert_array_equal(shown.data.ctrl, reference.data.ctrl)
    assert shown.data.time == reference.data.time == pytest.approx(0.012)
    assert viewer.sync_calls == 9 and viewer.closed


def test_passive_viewer_failure_retains_phase_and_lifecycle_state():
    sim, controller = make_simulator()

    def fail(state, target):
        raise RuntimeError("controller failed")

    controller.compute = fail
    manager = manager_with(sim)
    with (
        patch("mujoco.viewer.launch_passive", return_value=PassiveViewer(1)),
        pytest.raises(RuntimeError, match="controller failed"),
    ):
        manager.show("simulator")
    assert sim._state.get_state() == "viewing"


def test_successful_nested_control_returns_to_its_lifecycle_parent():
    sim, _ = make_simulator()
    robot = sim.robots["arm"]

    sim.step()
    assert sim._state.get_state() == "idle"
    robot.control()
    assert sim._state.get_state() == "idle"

    def check_viewing():
        assert sim._state.get_state() == "viewing"

    manager = manager_with(sim)
    viewer = PassiveViewer(1, on_lock=check_viewing)
    with patch("mujoco.viewer.launch_passive", return_value=viewer):
        manager.show("simulator")
    assert sim._state.get_state() == "idle"
    sim.reset()
    assert sim._state.get_state() == "idle"


def test_viewing_blocks_same_simulator_operations_but_not_independent_headless_work():
    sim, controller = make_simulator()
    manager = manager_with(sim, "first")
    other = Simulator(create_environment("empty"))
    manager.add_simulator("other", other)

    def inspect_open_scope():
        model = sim.model
        timestep = model.opt.timestep
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        data = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, sim.data, data, state_spec)
        controller_state = (controller.times.copy(), controller.tracking_error)
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            sim.reset()
        assert sim.model is model and sim.model.opt.timestep == timestep
        current = np.empty_like(data)
        mujoco.mj_getState(model, sim.data, current, state_spec)
        np.testing.assert_array_equal(current, data)
        assert (controller.times, controller.tracking_error) == controller_state

        sim._stop_requested = True
        operations = [
            sim.step,
            lambda: manager.show("first"),
            lambda: manager.save_frame("first"),
        ]
        for operation in operations:
            with pytest.raises(RuntimeError, match="Invalid state transition"):
                operation()
            assert sim._stop_requested
        assert other.step() == {}
        other.reset()

    viewer = PassiveViewer(1, on_lock=inspect_open_scope)
    with patch("mujoco.viewer.launch_passive", return_value=viewer):
        manager.show("first")
    assert sim._state.get_state() == "idle"


def test_passive_viewer_controls_after_gui_clock_rewind():
    sim, controller = make_simulator()
    manager = manager_with(sim)
    viewer = None

    def rewind_after_three_steps():
        if viewer.sync_calls == 3:
            mujoco.mj_resetData(sim.model, sim.data)

    viewer = PassiveViewer(4, on_sync=rewind_after_three_steps)
    with patch("mujoco.viewer.launch_passive", return_value=viewer):
        manager.show("simulator")
    np.testing.assert_allclose(controller.times, [0, 0.001, 0.002, 0])


def test_passive_viewer_keeps_configured_timestep_after_gui_edit():
    sim, controller = make_simulator()
    manager = manager_with(sim)

    def edit_timestep():
        sim.model.opt.timestep = 0.002

    viewer = PassiveViewer(5, on_sync=edit_timestep)
    with patch("mujoco.viewer.launch_passive", return_value=viewer):
        manager.show("simulator")
    assert sim.model.opt.timestep == sim.dt == 0.001
    assert sim.data.time == pytest.approx(0.005)
    np.testing.assert_allclose(controller.times, np.arange(5) * 0.001)


def test_passive_viewer_sync_failure_restores_timestep_and_closes_viewer():
    sim = Simulator(create_environment("empty"), dt=0.001)
    manager = manager_with(sim)

    def fail_after_timestep_edit():
        sim.model.opt.timestep = 0.002
        raise RuntimeError("sync failed")

    viewer = PassiveViewer(1, on_sync=fail_after_timestep_edit)
    with (
        patch("mujoco.viewer.launch_passive", return_value=viewer),
        pytest.raises(RuntimeError, match="sync failed"),
    ):
        manager.show("simulator")
    assert sim.model.opt.timestep == sim.dt == 0.001
    assert sim._state.get_state() == "viewing"
    assert viewer.closed


def test_passive_viewer_pacing_sleeps_only_for_positive_remainder():
    sim = Simulator(create_environment("empty"), dt=0.001)
    manager = manager_with(sim)
    viewer = PassiveViewer(2)
    with (
        patch("mujoco.viewer.launch_passive", return_value=viewer),
        patch(
            "mujoco_lab.simulator_manager.time.monotonic",
            side_effect=[0.0, 0.00025, 1.0, 1.002],
        ),
        patch("mujoco_lab.simulator_manager.time.sleep") as sleep,
    ):
        manager.show("simulator")
    sleep.assert_called_once_with(pytest.approx(0.00075))


@pytest.mark.parametrize(
    "timing",
    [
        {"dt": 0},
        {"dt": -0.001},
        {"dt": float("nan")},
        {"dt": float("inf")},
    ],
)
def test_invalid_periods_are_rejected_without_changing_the_model(timing):
    scene = mujoco.MjSpec.from_string("<mujoco/>")
    original = scene.option.timestep
    with pytest.raises(ValueError):
        Simulator(scene, **timing)
    assert scene.option.timestep == original


def test_cli_accepts_single_period(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--robot",
            "forte",
            "--controller",
            "pd",
            "--dt",
            "0.001",
            "--steps",
            "10",
        ],
        cwd=tmp_path,
        env={**os.environ, "MUJOCO_GL": "disable"},
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_rejects_invalid_simulator_period(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--robot",
            "forte",
            "--dt",
            "0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "dt must be finite and positive" in result.stderr
