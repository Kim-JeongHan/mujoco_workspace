import mujoco
import numpy as np
import pytest

from mujoco_lab import ROBOT_NAMES, RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_composition_preserves_robot_frames_controls_and_single_floor(name):
    baseline = Simulator(
        create_environment("empty"), robots=[RobotSpec(name, name, config=load_robot_config(name))]
    )
    sim = Simulator(
        create_environment("warehouse"),
        robots=[RobotSpec(name, name, config=load_robot_config(name))],
    )
    original, original_data = baseline.model, baseline.data
    model, data = sim.model, sim.data
    assert np.count_nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
    assert model.nq == original.nq
    assert model.nu == original.nu
    assert model.neq == original.neq
    np.testing.assert_allclose(data.qpos, original_data.qpos)
    np.testing.assert_allclose(data.ctrl, original_data.ctrl)
    for index in range(1, original.nbody):
        body_name = original.body(index).name
        np.testing.assert_allclose(
            data.body(body_name).xpos,
            original_data.body(body_name).xpos + [0, 0, 0.8],
            atol=1e-12,
        )
        np.testing.assert_allclose(data.body(body_name).xmat, original_data.body(body_name).xmat)


def test_shelf_opening_is_free_but_shelf_board_collides():
    spec = create_environment("warehouse")
    probe = spec.worldbody.add_body(name="probe", pos=[0.4, 0.55, 0.93])
    probe.add_freejoint()
    probe.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=0.1)
    simulator = Simulator(spec)
    model, data = simulator.model, simulator.data
    assert data.ncon == 0
    data.qpos[2] = 0.8275
    mujoco.mj_forward(model, data)
    contacted = {model.geom(int(index)).name for contact in data.contact for index in contact.geom}
    assert "probe" in contacted
    assert "large_shelf/shelf_1" in contacted
