import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from mujoco_lab import ROBOT_NAMES, RobotSpec, Simulator, create_environment
from mujoco_lab.assets import ASSET_PATH

ASSETS = ASSET_PATH / "environment"


@pytest.mark.parametrize(
    "name,size,count",
    [
        ("table", [0.80, 1.20, 0.80], 1),
        ("large_shelf", [0.80, 0.30, 2.05], 10),
        ("small_shelf", [0.80, 0.28, 1.06], 1),
    ],
)
def test_object_bounds_match_expected_dimensions(name, size, count):
    simulator = Simulator(mujoco.MjSpec.from_file(str(ASSETS / "objects" / name / "model.xml")))
    model, data = simulator.model, simulator.data
    assert model.nq == 0
    assert model.ngeom == count
    assert np.all(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    lower = np.min(data.geom_xpos - model.geom_size, axis=0)
    upper = np.max(data.geom_xpos + model.geom_size, axis=0)
    np.testing.assert_allclose(upper - lower, size, atol=1e-12)
    np.testing.assert_allclose(lower, [-size[0] / 2, -size[1] / 2, 0], atol=1e-12)


@pytest.mark.parametrize("environment,count", [("table_shelf", 12), ("warehouse", 13)])
def test_furnished_layout_supports_mount_and_preserves_shelves(environment, count):
    simulator = Simulator(create_environment(environment))
    model, data = simulator.model, simulator.data
    assert model.nq == 0
    assert model.ngeom == count
    mount = data.site("robot_mount").xpos
    np.testing.assert_allclose(mount, [0, 0, 0.8])
    np.testing.assert_allclose(data.geom("table/box").xpos - mount, [0.35, 0, -0.4])
    np.testing.assert_allclose(
        data.geom("table/box").xmat.reshape(3, 3),
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data.geom("large_shelf/left_panel").xpos - mount,
        [0.01, 0.57, 0.225],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data.geom("large_shelf/shelf_1").xpos - mount,
        [0.4, 0.57, 0.0275],
        atol=1e-12,
    )
    table = model.geom("table/box")
    table_center = data.geom("table/box").xpos
    table_half_extents = np.abs(data.geom("table/box").xmat.reshape(3, 3)) @ table.size
    np.testing.assert_allclose(table_center + table_half_extents, [0.95, 0.4, 0.8])
    np.testing.assert_allclose(table_center - table_half_extents, [-0.25, -0.4, 0])
    assert table_center[1] + table_half_extents[1] < 0.57 - 0.14
    if environment == "warehouse":
        np.testing.assert_allclose(
            data.geom("small_shelf/box").xpos - mount,
            [0.4, -0.6, -0.27],
            atol=1e-12,
        )
        assert table_center[1] - table_half_extents[1] > -0.6 + 0.14


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_composition_preserves_robot_frames_controls_and_single_floor(name):
    baseline = Simulator(create_environment("empty"), robots=[RobotSpec(name, name)])
    sim = Simulator(create_environment("warehouse"), robots=[RobotSpec(name, name)])
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


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_furnished_table_supports_bundled_robot_base(name):
    sim = Simulator(create_environment("warehouse"), robots=[RobotSpec(name, name)])
    model, data = sim.model, sim.data
    table = model.geom("table/box")
    center = data.geom("table/box").xpos
    half_extents = np.abs(data.geom("table/box").xmat.reshape(3, 3)) @ table.size
    base_body = (
        model.body(name + "/base_link_inertia").id
        if name.startswith("ur")
        else sim.robots[name].state.root_body_id
    )
    base_vertices = []
    for geom_id in np.flatnonzero(model.geom_bodyid == base_body):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        base_vertices.append(vertices @ rotation.T + data.geom_xpos[geom_id])
    bounds = np.concatenate(base_vertices)
    assert np.all(bounds[:, :2] >= center[:2] - half_extents[:2] + 0.01)
    assert np.all(bounds[:, :2] <= center[:2] + half_extents[:2] - 0.01)
    assert bounds[:, 2].min() >= center[2] + half_extents[2] - 1e-4
    assert data.ncon == 0


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


@pytest.mark.parametrize("robot", [None, "panda"])
def test_environment_cli_runs_outside_project_directory(robot, tmp_path):
    command = [
        sys.executable,
        "-m",
        "mujoco_lab",
        "--command",
        "simulate",
        "--environment",
        "warehouse",
        "--steps",
        "100",
    ]
    if robot is not None:
        command += ["--robot", robot]
    env = {**os.environ, "MUJOCO_GL": "disable"}
    env.pop("DISPLAY", None)
    subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=True)


def test_unknown_environment_is_rejected():
    with pytest.raises(KeyError, match="missing"):
        create_environment("missing")
