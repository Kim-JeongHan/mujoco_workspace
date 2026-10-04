"""Video resources close in order and restore model settings on every exit path."""

from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import pytest

from mujoco_lab import Simulator
from mujoco_lab.rendering.video import VideoRecorder


@pytest.mark.parametrize(
    "stage,error_type,cleanup_failure",
    [
        (None, RuntimeError, False),
        ("renderer_init", RuntimeError, False),
        ("writer_init", RuntimeError, False),
        ("body", RuntimeError, False),
        ("writer_close", RuntimeError, False),
        ("renderer_close", RuntimeError, False),
        ("writer_init", KeyboardInterrupt, False),
        ("body", KeyboardInterrupt, False),
        ("writer_init", RuntimeError, True),
    ],
)
def test_recorder_releases_acquired_resources_and_restores_size(
    stage, error_type, cleanup_failure, tmp_path, monkeypatch
):
    simulator = Simulator(mujoco.MjSpec.from_string("<mujoco/>"))
    settings = simulator.model.vis.global_
    settings.offwidth, settings.offheight = 32, 24
    closed = []
    failure = error_type("recording failed")
    renderer_error = RuntimeError("renderer close failed")

    def close_renderer():
        assert (settings.offwidth, settings.offheight) == (64, 48)
        closed.append("renderer")
        if cleanup_failure:
            raise renderer_error
        if stage == "renderer_close":
            raise failure

    def close_writer():
        assert (settings.offwidth, settings.offheight) == (64, 48)
        closed.append("writer")
        if stage == "writer_close":
            raise failure

    def make_renderer(model, *, width, height):
        assert model is simulator.model
        assert (settings.offwidth, settings.offheight) == (width, height) == (64, 48)
        if stage == "renderer_init":
            raise failure
        return SimpleNamespace(close=close_renderer)

    def make_writer(*args):
        if stage == "writer_init":
            raise failure
        return SimpleNamespace(close=close_writer)

    monkeypatch.setattr("mujoco_lab.rendering.video.mujoco.Renderer", make_renderer)
    monkeypatch.setattr("mujoco_lab.rendering.video.VideoWriter", make_writer)
    expected_error = renderer_error if cleanup_failure else failure
    with (
        pytest.raises(type(expected_error)) if stage is not None else nullcontext() as raised,
        VideoRecorder(simulator, tmp_path / "video.mp4", width=64, height=48),
    ):
        if stage == "body":
            raise failure
    if stage is not None:
        assert raised.value is expected_error
    if cleanup_failure:
        assert renderer_error.__context__ is failure
    expected = (
        []
        if stage == "renderer_init"
        else ["renderer"]
        if stage == "writer_init"
        else ["writer", "renderer"]
    )
    assert closed == expected
    assert (settings.offwidth, settings.offheight) == (32, 24)
