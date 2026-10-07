from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from simulator_helpers import PassiveViewer, manager_with

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import Controller


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
        robots=[RobotSpec("arm", "forte", config=load_robot_config("forte"))],
        dt=0.001,
    )
    controller = RecordingController()
    sim.robots["arm"].change_controller(controller)
    return sim, controller


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


def test_viewing_blocks_same_simulator_operations_but_not_independent_headless_work():
    sim, controller = make_simulator()
    manager = manager_with(sim, "first")
    other = Simulator(create_environment("empty"))
    manager.add_simulator("other", other)

    def inspect_open_scope():
        assert sim._state.get_state() == "viewing"
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
    assert sim._state.get_state() == "idle"
    assert viewer.closed


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
