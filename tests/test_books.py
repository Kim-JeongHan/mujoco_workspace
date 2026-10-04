"""Verify book physics, shelf clearance, and robot scene composition."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import BOOK_TYPES, RobotSpec, Simulator, create_book_insertion, create_environment
from mujoco_lab.assets import BOOK_SCENES, ENVIRONMENT_SCENES
from mujoco_lab.assets.loader import load_robot_config

# Expected physics retained from the original presets, independent of XML parsing.
BOOK_PHYSICS = {
    "small": ((0.125, 0.020, 0.190), 0.25),
    "medium": ((0.156, 0.034, 0.236), 0.45),
    "thick": ((0.156, 0.050, 0.236), 0.65),
    "large": ((0.190, 0.040, 0.280), 0.75),
}


@pytest.mark.parametrize("book_type", BOOK_TYPES)
def test_book_frames_and_separate_visual_collision(book_type):
    size, mass = BOOK_PHYSICS[book_type]
    sim = Simulator(create_book_insertion(book_type))
    model, data = sim.model, sim.data
    body = model.body("book")
    collision = model.geom("book/collision")
    visual = model.geom("book/visual")
    assert model.nq == 7
    assert model.nv == 6
    np.testing.assert_allclose(body.mass, mass)
    np.testing.assert_allclose(body.ipos, [0, 0, 0])
    np.testing.assert_allclose(body.iquat, [1, 0, 0, 0])
    np.testing.assert_allclose(
        data.body("book").xmat.reshape(3, 3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12
    )
    np.testing.assert_allclose(data.body("book").xipos, data.body("book").xpos)
    np.testing.assert_allclose(2 * collision.size, size)
    np.testing.assert_allclose(collision.pos, [0, 0, 0])
    np.testing.assert_allclose(collision.friction, [0.9, 0.005, 0.0001])
    assert collision.type == mujoco.mjtGeom.mjGEOM_BOX
    assert collision.contype == 1 and collision.conaffinity == 1
    assert visual.contype == 0 and visual.conaffinity == 0
    np.testing.assert_allclose(visual.size, collision.size)
    assert model.body("large_shelf/base").jntnum == 0
    assert model.body("large_shelf/base").dofnum == 0
    shelf_collisions = [
        g
        for g in range(model.ngeom)
        if model.geom(g).name.startswith("large_shelf/") and model.geom_contype[g] != 0
    ]
    assert len(shelf_collisions) == 10
    assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in shelf_collisions)


def test_medium_inertia_matches_full_dimensions_and_visual_does_not_add_mass():
    scene = create_book_insertion()
    # Arbitrarily enlarge the visual: collision and physical inertia stay fixed.
    scene.geom("book/visual").size = [1, 1, 1]
    sim = Simulator(scene)
    np.testing.assert_allclose(sim.model.body("book").mass, 0.45)
    np.testing.assert_allclose(
        sim.model.body("book").inertia, [0.00213195, 0.0030012, 0.00095595], atol=1e-12
    )


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


def test_book_shelf_clearance_and_original_cube_furniture_are_preserved():
    baseline = Simulator(create_environment("table_shelf"))
    sim = Simulator(create_environment("book_shelf"))
    for name in (
        "table/box",
        "large_shelf/left_panel",
        "large_shelf/right_panel",
        "large_shelf/back_panel",
        "large_shelf/bottom",
        "large_shelf/top",
        "large_shelf/shelf_1",
    ):
        np.testing.assert_allclose(sim.data.geom(name).xpos, baseline.data.geom(name).xpos)
        np.testing.assert_allclose(sim.model.geom(name).size, baseline.model.geom(name).size)
    lower = sim.data.geom("large_shelf/shelf_1").xpos[2] + 0.0075
    upper = sim.data.geom("large_shelf/shelf_2").xpos[2] - 0.0075
    assert lower == pytest.approx(0.835)
    assert upper - lower == pytest.approx(0.34)
    assert upper - lower - BOOK_PHYSICS["medium"][0][2] == pytest.approx(0.104)
    assert upper - lower - BOOK_PHYSICS["large"][0][2] == pytest.approx(0.06)
    # Loading the original scene again must retain its 215 mm first opening.
    original = Simulator(create_environment("table_shelf"))
    assert original.model.geom("large_shelf/shelf_2").pos[2] == pytest.approx(1.0575)
    for g in sim.model.geom_bodyid.nonzero()[0]:
        geom = sim.model.geom(int(g))
        if geom.name.startswith("large_shelf/") and not geom.name.endswith("_visual"):
            visual = sim.model.geom(f"{geom.name}_visual")
            np.testing.assert_allclose(visual.pos, geom.pos)
            np.testing.assert_allclose(visual.size, geom.size)
            assert visual.contype == 0 and visual.conaffinity == 0


@pytest.mark.parametrize("robot_type", ["forte", "panda"])
def test_book_scene_composes_with_robot_and_preserves_free_book(robot_type):
    sim = Simulator(
        create_book_insertion(),
        robots=[RobotSpec(robot_type, robot_type, config=load_robot_config(robot_type))],
    )
    assert sim.model.joint("book/free_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert sim.model.body("book").jntnum == 1
    assert np.count_nonzero(sim.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
    np.testing.assert_allclose(sim.data.site("robot_mount").xpos, [0, 0, 0.8])
    book_id = sim.model.body("book").id
    for contact in sim.data.contact:
        if book_id in sim.model.geom_bodyid[contact.geom]:
            names = {sim.model.geom(int(g)).name for g in contact.geom}
            assert names == {"book/collision", "table/box"}


@pytest.mark.parametrize("book_type", BOOK_TYPES)
def test_xml_inertia_matches_uniform_box(book_type):
    sim = Simulator(create_book_insertion(book_type))
    (x, y, z), mass = BOOK_PHYSICS[book_type]
    expected = mass / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    np.testing.assert_allclose(sim.model.body("book").inertia, expected, atol=1e-12)


@pytest.mark.parametrize("book_type", BOOK_TYPES)
def test_book_xml_loads_directly_with_matching_placement_and_goals(book_type):
    # Loading XML alone must provide the complete scene without Python adjustments.
    model = mujoco.MjModel.from_xml_path(str(BOOK_SCENES[book_type]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    size, _ = BOOK_PHYSICS[book_type]
    np.testing.assert_allclose(data.body("book").xpos, [0.48, 0.2, 0.8 + size[2] / 2])
    np.testing.assert_allclose(data.site("book_target").xpos, [0.4, 0.57, 0.835 + size[2] / 2])
    np.testing.assert_allclose(2 * model.site("book_target").size, size)
    rotation = data.body("book").xmat.reshape(3, 3)
    target_rotation = data.site("book_target").xmat.reshape(3, 3)
    np.testing.assert_allclose(rotation, target_rotation, atol=1e-12)
    np.testing.assert_allclose(
        data.site("book_grasp_goal").xpos - data.site("book_target").xpos,
        data.site("book/grasp_target").xpos - data.body("book").xpos,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data.site("book_grasp_goal").xmat, data.site("book/grasp_target").xmat, atol=1e-12
    )
    np.testing.assert_allclose(model.geom("large_shelf/shelf_2").pos[2], 1.1825)


def test_furniture_xml_loads_directly_with_book_shelf_clearance():
    model = mujoco.MjModel.from_xml_path(str(ENVIRONMENT_SCENES["book_shelf"]))
    assert model.nq == 0
    lower = model.geom("large_shelf/shelf_1")
    upper = model.geom("large_shelf/shelf_2")
    opening = upper.pos[2] - upper.size[2] - lower.pos[2] - lower.size[2]
    assert opening == pytest.approx(0.34)
