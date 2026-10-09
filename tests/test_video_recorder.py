"""Video cleanup after normal use, initialization failure, and interrupted recording."""

from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import pytest

from mujoco_lab import Simulator
from mujoco_lab.rendering.video import VideoRecorder


@pytest.mark.parametrize(
    "stage,error_type",
    [
        (None, RuntimeError),
        ("renderer_init", RuntimeError),
        ("writer_init", RuntimeError),
        ("body", RuntimeError),
        ("body", KeyboardInterrupt),
    ],
)
def test_recorder_releases_acquired_resources_and_restores_size(
    stage, error_type, tmp_path, monkeypatch
):
    simulator = Simulator(mujoco.MjSpec.from_string("<mujoco/>"))
    settings = simulator.model.vis.global_
    settings.offwidth, settings.offheight = 32, 24
    closed = []
    failure = error_type("recording failed")

    def close_renderer():
        assert (settings.offwidth, settings.offheight) == (64, 48)
        closed.append("renderer")

    def close_writer():
        assert (settings.offwidth, settings.offheight) == (64, 48)
        closed.append("writer")

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
    with (
        pytest.raises(error_type) if stage is not None else nullcontext() as raised,
        VideoRecorder(simulator, tmp_path / "video.mp4", width=64, height=48),
    ):
        if stage == "body":
            raise failure
    if stage is not None:
        assert raised.value is failure
    expected = (
        []
        if stage == "renderer_init"
        else ["renderer"]
        if stage == "writer_init"
        else ["writer", "renderer"]
    )
    assert closed == expected
    assert (settings.offwidth, settings.offheight) == (32, 24)
