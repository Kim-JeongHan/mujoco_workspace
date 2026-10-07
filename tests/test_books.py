"""Verify book physics, shelf clearance, and robot scene composition."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import BOOK_TYPES, Simulator, create_book_insertion

# Expected physics retained from the original presets, independent of XML parsing.
BOOK_PHYSICS = {
    "small": ((0.125, 0.020, 0.190), 0.25),
    "medium": ((0.156, 0.034, 0.236), 0.45),
    "thick": ((0.156, 0.050, 0.236), 0.65),
    "large": ((0.190, 0.040, 0.280), 0.75),
}


@pytest.mark.parametrize("book_type", BOOK_TYPES)
def test_book_settles_on_table_and_inside_shelf(book_type):
    sim = Simulator(create_book_insertion(book_type))
    sim.run_steps(1000)
    half_height = BOOK_PHYSICS[book_type][0][2] / 2
    np.testing.assert_allclose(
        sim.data.body("book").xpos, [0.48, 0.2, 0.8 + half_height], atol=0.0002
    )
    sim.reset()
    address = int(sim.model.joint("book/free_joint").qposadr[0])
    target = sim.data.site("book_target").xpos.copy()
    sim.data.qpos[address : address + 3] = target
    mujoco.mj_forward(sim.model, sim.data)
    sim.run_steps(1000)
    np.testing.assert_allclose(sim.data.body("book").xpos, target, atol=0.0002)
    assert np.linalg.norm(sim.data.qvel) < 1e-5
    pairs = {frozenset(sim.model.geom(int(g)).name for g in c.geom) for c in sim.data.contact}
    assert frozenset(("book/collision", "large_shelf/shelf_1")) in pairs
    assert all("book/visual" not in pair for pair in pairs)


@pytest.mark.parametrize("book_type", BOOK_TYPES)
def test_front_insertion_path_is_clear_and_side_panel_blocks_book(book_type):
    sim = Simulator(create_book_insertion(book_type))
    address = int(sim.model.joint("book/free_joint").qposadr[0])
    target = sim.data.site("book_target").xpos.copy()
    target[2] += 0.001  # Lift 1 mm above the shelf board for a clearance probe.
    for y in np.linspace(0.25, target[1], 20):
        sim.data.qpos[address : address + 3] = [target[0], y, target[2]]
        mujoco.mj_forward(sim.model, sim.data)
        assert sim.data.ncon == 0
    # Crossing the +X side wall must generate physical box contact.
    sim.data.qpos[address] = 0.79
    mujoco.mj_forward(sim.model, sim.data)
    pairs = {frozenset(sim.model.geom(int(g)).name for g in c.geom) for c in sim.data.contact}
    assert frozenset(("book/collision", "large_shelf/right_panel")) in pairs


@pytest.mark.parametrize("book_type", BOOK_TYPES)
def test_xml_inertia_matches_uniform_box(book_type):
    sim = Simulator(create_book_insertion(book_type))
    (x, y, z), mass = BOOK_PHYSICS[book_type]
    expected = mass / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    np.testing.assert_allclose(sim.model.body("book").inertia, expected, atol=1e-12)
