import io
from contextlib import redirect_stderr
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


def test_repeated_manager_access_does_not_reset_the_registry_or_logger():
    manager = SimulatorManager.get_instance()
    logger = manager.logger
    handler_count = len(logger._logger.handlers)
    simulator = make_empty_simulator()
    name = "manager-test-viewer"
    manager.add_simulator(name, simulator)
    viewer = Mock()
    viewer._sim = lambda: None

    def inspect_manager():
        repeated = SimulatorManager.get_instance()
        assert repeated.simulators[name] is simulator
        assert repeated.logger is logger
        assert len(repeated.logger._logger.handlers) == handler_count
        return False

    viewer.is_running.side_effect = inspect_manager
    try:
        with patch("mujoco.viewer.launch_passive", return_value=viewer):
            manager.show(name)
    finally:
        manager.simulators.pop(name, None)


def test_shared_logger_uses_the_current_standard_error_stream():
    logger = SimulatorManager.get_instance().logger
    stream = io.StringIO()

    with redirect_stderr(stream):
        logger.info("shared logger output")

    assert "[INFO] shared logger output" in stream.getvalue()


def test_manager_registers_multiple_simulators_without_overwriting_names():
    manager = SimulatorManager.get_instance()
    first = make_empty_simulator()
    second = make_empty_simulator()
    names = ("manager-test-first", "manager-test-second")

    try:
        manager.add_simulator(names[0], first)
        manager.add_simulator(names[1], second)

        assert manager.simulators[names[0]] is first
        assert manager.simulators[names[1]] is second
        with pytest.raises(KeyError, match=names[0]):
            manager.add_simulator(names[0], second)
        assert manager.simulators[names[0]] is first
    finally:
        for name in names:
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


def test_direct_managers_keep_registries_independent():
    first_manager = SimulatorManager()
    second_manager = SimulatorManager()
    first = make_empty_simulator()
    second = make_empty_simulator()
    first_manager.add_simulator("shared-name", first)
    second_manager.add_simulator("shared-name", second)
    assert first_manager.simulators["shared-name"] is first
    assert second_manager.simulators["shared-name"] is second


def test_viewer_launch_failure_retains_viewing_state():
    simulator = make_empty_simulator()
    manager = SimulatorManager()
    manager.add_simulator("failed", simulator)

    with (
        patch("mujoco.viewer.launch_passive", side_effect=RuntimeError("launch failed")),
        pytest.raises(RuntimeError, match="launch failed"),
    ):
        manager.show("failed")

    assert simulator._state.get_state() == "viewing"


def test_viewer_close_failure_retains_viewing_state():
    simulator = make_empty_simulator()
    manager = SimulatorManager()
    manager.add_simulator("failed", simulator)
    viewer = Mock()
    viewer.is_running.return_value = False
    viewer.close.side_effect = RuntimeError("close failed")

    with (
        patch("mujoco.viewer.launch_passive", return_value=viewer),
        pytest.raises(RuntimeError, match="close failed"),
    ):
        manager.show("failed")

    assert simulator._state.get_state() == "viewing"


class FakeRenderer:
    created_models = []
    updated_data = []

    def __init__(self, model, *, height, width):
        assert (height, width) == (480, 640)
        self.created_models.append(model)
        self.scene = object()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def update_scene(self, data):
        self.updated_data.append(data)

    def render(self):
        return np.zeros((2, 3, 3), dtype=np.uint8)


def test_manager_renders_selected_names_to_default_and_explicit_outputs(tmp_path, monkeypatch):
    first = Simulator(mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>"))
    second = Simulator(
        mujoco.MjSpec.from_string(
            "<mujoco><worldbody><body><joint/><geom size='0.1'/></body></worldbody></mujoco>"
        )
    )
    manager = SimulatorManager()
    manager.add_simulator("first", first)
    manager.add_simulator("second", second)
    FakeRenderer.created_models = []
    FakeRenderer.updated_data = []
    explicit = tmp_path / "second.png"
    monkeypatch.chdir(tmp_path)

    with (
        patch("mujoco.Renderer", FakeRenderer),
        patch("mujoco_lab.rendering.annotations.annotate") as annotate,
    ):
        default = manager.save_frame("first")
        selected = manager.save_frame("second", explicit)

    assert default == (tmp_path / "frame.png").resolve()
    assert selected == explicit.resolve()
    assert default.is_file() and selected.is_file()
    assert FakeRenderer.created_models == [first.model, second.model]
    assert [data.qpos.size for data in FakeRenderer.updated_data] == [0, 1]
    assert [call.args[1] for call in annotate.call_args_list] == [first, second]
    assert first._state.get_state() == second._state.get_state() == "idle"


def test_viewer_draw_hook_owns_user_scene_and_runs_under_lock():
    simulator = make_empty_simulator()
    manager = SimulatorManager()
    manager.add_simulator("drawn", simulator)
    viewer = Mock()
    viewer._sim = lambda: None
    viewer.is_running.side_effect = [True, True, False]
    viewer.user_scn = mujoco.MjvScene(simulator.model, 4)
    viewer.cam = mujoco.MjvCamera()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.4, -0.2, 0.7]
    camera.distance = 1.6
    camera.azimuth = 132
    camera.elevation = -18
    inside_lock = False
    seen = []

    class Lock:
        def __enter__(self):
            nonlocal inside_lock
            inside_lock = True

        def __exit__(self, *_):
            nonlocal inside_lock
            inside_lock = False

    viewer.lock.side_effect = Lock

    def draw(scene, data):
        assert inside_lock
        assert scene is viewer.user_scn
        assert scene.ngeom == 0
        assert data is simulator.data
        seen.append(data.time)
        scene.ngeom = 1

    with patch("mujoco.viewer.launch_passive", return_value=viewer):
        manager.show("drawn", draw=draw, camera=camera)
    assert len(seen) == 2
    assert seen[0] > 0 and seen[1] > seen[0]
    assert viewer.sync.call_count == 2
    np.testing.assert_array_equal(viewer.cam.lookat, camera.lookat)
    assert viewer.cam.distance == camera.distance
    assert viewer.cam.azimuth == camera.azimuth
    assert viewer.cam.elevation == camera.elevation
    assert simulator._state.get_state() == "idle"


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


@pytest.mark.parametrize("operation", ["show", "save_frame"])
def test_unknown_rendering_name_does_not_mutate_resources(operation, tmp_path):
    manager = SimulatorManager()

    with pytest.raises(KeyError, match="missing"):
        if operation == "show":
            manager.show("missing")
        else:
            manager.save_frame("missing", tmp_path / "unused.png")

    assert manager.simulators == {}
    assert not (tmp_path / "unused.png").exists()


def test_rendering_methods_are_manager_only():
    simulator = make_empty_simulator()

    assert not hasattr(simulator, "show")
    assert not hasattr(simulator, "save_frame")
