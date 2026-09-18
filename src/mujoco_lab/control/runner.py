"""Apply controller torques and advance native MuJoCo simulation."""

import numpy as np

from mujoco_lab.control.stats import RunStats
from mujoco_lab.state import read_state


def apply_control(model, data, controller, stats: RunStats | None = None) -> None:
    lower = model.actuator_ctrlrange[:, 0]
    upper = model.actuator_ctrlrange[:, 1]
    torques = np.asarray(controller.torques(read_state(data)), dtype=float)
    clipped = np.clip(torques, lower, upper)
    data.ctrl[:] = clipped
    error = getattr(controller, "tracking_error", None)
    if stats is not None:
        stats.update(bool(np.any(clipped != torques)), error)


def run_steps(model, data, steps: int, controller=None) -> RunStats:
    """Advance exactly ``steps`` physics steps with optional Forte feedback control."""
    import mujoco

    if steps < 0:
        raise ValueError("steps must be zero or greater")
    stats = RunStats()
    if controller is None:
        mujoco.mj_step(model, data, nstep=steps)
        stats.steps = steps
    else:
        for _ in range(steps):
            apply_control(model, data, controller, stats)
            mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Simulation diverged")
    return stats
