from contextlib import nullcontext
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import pytest

from mujoco_lab import Simulator, SimulatorManager


def make_empty_simulator():
    return Simulator(mujoco.MjSpec.from_string("<mujoco/>"))


def test_simulator_construction_preserves_the_shared_manager_registry_and_logger():
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    first = make_empty_simulator()
    name = "manager-test-construction"
    manager.add_simulator(name, first)

    try:
        registry = manager.simulators.copy()
        second = make_empty_simulator()

        assert SimulatorManager.get_instance() is manager
        assert manager.logger is logger
        assert manager.simulators == registry
        assert manager.simulators[name] is first
        assert second not in manager.simulators.values()
    finally:
        manager.simulators.pop(name, None)


def test_manager_removes_and_replaces_only_existing_registration():
    manager = SimulatorManager.get_instance()
    first = make_empty_simulator()
    replacement = make_empty_simulator()
    unrelated = make_empty_simulator()
    names = ("manager-test-target", "manager-test-unrelated")

    try:
        manager.add_simulator(names[0], first)
        manager.add_simulator(names[1], unrelated)

        assert manager.replace_simulator(names[0], replacement) is first
        assert manager.simulators[names[0]] is replacement
        assert manager.simulators[names[1]] is unrelated
        assert SimulatorManager.get_instance() is manager
        registry = manager.simulators.copy()
        make_empty_simulator()
        assert SimulatorManager.get_instance().simulators == registry
        assert manager.remove_simulator(names[0]) is replacement
        assert manager.simulators[names[1]] is unrelated
        with pytest.raises(KeyError, match=names[0]):
            manager.remove_simulator(names[0])
        with pytest.raises(KeyError, match=names[0]):
            manager.replace_simulator(names[0], first)
    finally:
        for name in names:
            manager.simulators.pop(name, None)


@pytest.mark.parametrize("failure", [False, True])
def test_recording_closes_before_viewer_and_restores_idle_state(failure, tmp_path):
    simulator = make_empty_simulator()
    manager = SimulatorManager()
    manager.add_simulator("recording", simulator)
    events = []
    viewer = Mock()
    viewer._sim = lambda: None
    viewer.cam = mujoco.MjvCamera()
    viewer.lock.side_effect = nullcontext
    viewer.is_running.side_effect = [True, False]
    viewer.close.side_effect = lambda: events.append("viewer closed")

    class Recorder:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            assert simulator._state.get_state() == "viewing"
            events.append("recorder closed")

        def record_initial(self, camera):
            events.append("initial frame")

        def record_due(self, camera):
            events.append("physics frame")
            if failure:
                raise RuntimeError("recording failed")

    def launch(model, data, *, key_callback):
        key_callback(32)
        return viewer

    with (
        patch("mujoco.viewer.launch_passive", side_effect=launch),
        patch("mujoco_lab.simulator_manager.VideoRecorder", Recorder),
        pytest.raises(RuntimeError, match="recording failed") if failure else nullcontext(),
    ):
        manager.show("recording", video=tmp_path / "recording.mp4")

    assert events == ["initial frame", "physics frame", "recorder closed", "viewer closed"]
    assert simulator._state.get_state() == "idle"
    assert not simulator._stop_requested
    simulator.reset()


class FakeRenderer:
    def __init__(self, model, *, height, width):
        self.scene = object()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def update_scene(self, data):
        pass

    def render(self):
        return np.zeros((2, 3, 3), dtype=np.uint8)


@pytest.mark.parametrize("stage", ["create", "render", "save", "close"])
def test_frame_failure_closes_renderer_and_releases_lifecycle(stage, tmp_path):
    simulator = make_empty_simulator()
    manager = SimulatorManager()
    manager.add_simulator("failed", simulator)
    failure = RuntimeError("frame failed")
    closed = []

    class Renderer(FakeRenderer):
        def __init__(self, *args, **kwargs):
            if stage == "create":
                raise failure
            super().__init__(*args, **kwargs)

        def render(self):
            if stage == "render":
                raise failure
            return super().render()

        def __exit__(self, *_):
            assert simulator._state.get_state() == "rendering"
            closed.append(True)
            if stage == "close":
                raise failure

    with (
        patch("mujoco.Renderer", Renderer),
        patch("mujoco_lab.rendering.annotations.annotate"),
        patch("PIL.Image.Image.save", side_effect=failure) if stage == "save" else nullcontext(),
        pytest.raises(RuntimeError, match="frame failed") as raised,
    ):
        manager.save_frame("failed", tmp_path / "failed.png")
    assert raised.value is failure
    assert closed == ([] if stage == "create" else [True])
    assert simulator._state.get_state() == "idle"
    assert not simulator._stop_requested
    simulator.reset()
    simulator.run_steps(1)


def test_save_frame_draw_hook_receives_copied_data(tmp_path):
    simulator = Simulator(
        mujoco.MjSpec.from_string(
            "<mujoco><worldbody><body><joint/><geom size='0.1'/></body></worldbody></mujoco>"
        )
    )
    manager = SimulatorManager()
    manager.add_simulator("drawn", simulator)
    before = simulator.data.qpos.copy()
    seen = []

    def draw(scene, data):
        assert scene is not None
        assert data is not simulator.data
        seen.append(data.time)
        data.qpos[:] = 0.3

    with (
        patch("mujoco.Renderer", FakeRenderer),
        patch("mujoco_lab.rendering.annotations.annotate"),
    ):
        manager.save_frame("drawn", tmp_path / "drawn.png", draw=draw)
    assert len(seen) == 1
    np.testing.assert_array_equal(simulator.data.qpos, before)
    assert simulator._state.get_state() == "idle"
