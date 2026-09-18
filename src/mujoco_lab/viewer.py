"""Interactive viewing and offscreen PNG rendering."""

from contextlib import contextmanager
from pathlib import Path

import mujoco

from mujoco_lab.control.runner import apply_control


def show(model: mujoco.MjModel, data: mujoco.MjData, controller=None) -> None:
    """Run the native viewer, which completes GUI cleanup before returning."""
    import mujoco.viewer

    with scoped_control_callback(model, data, controller):
        mujoco.viewer.launch(model, data)


def save_frame(
    model: mujoco.MjModel, data: mujoco.MjData, output: str | Path, controller=None
) -> Path:
    """Save the current scene as a 640 x 480 PNG and release the OpenGL context."""
    from PIL import Image

    from mujoco_lab.control.visualization import annotate

    path = Path(output).expanduser().resolve()
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        mujoco.mj_forward(model, data)
        if controller is not None:
            apply_control(model, data, controller)
            mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        if controller is not None:
            annotate(renderer.scene, model, data, controller)
        pixels = renderer.render()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path, format="PNG")
    return path


@contextmanager
def scoped_control_callback(model, data, controller):
    """Control the native viewer's physics loop and restore the host callback on exit."""
    if controller is None:
        yield
        return
    previous = mujoco.get_mjcb_control()

    def callback(active_model, active_data):
        if active_model is model and active_data is data:
            apply_control(model, data, controller)
        elif previous is not None:
            previous(active_model, active_data)

    mujoco.set_mjcb_control(callback)
    try:
        yield
    finally:
        mujoco.set_mjcb_control(previous)
