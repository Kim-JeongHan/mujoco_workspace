import json
import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from mujoco_lab import ROBOT_NAMES, create_environment, create_robot
from mujoco_lab.simulation import ASSET_ROOT, initialize_data

ASSETS = ASSET_ROOT


@pytest.mark.parametrize(
    "name,size,count",
    [
        ("table", [0.56, 0.90, 0.80], 1),
        ("large_shelf", [0.80, 0.30, 2.05], 10),
        ("small_shelf", [0.80, 0.28, 1.06], 1),
    ],
)
def test_object_bounds_match_source_dimensions(name, size, count):
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "objects" / name / "model.xml"))
    data = initialize_data(model)
    assert model.nq == 0
    assert model.ngeom == count
    assert np.all(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    lower = np.min(data.geom_xpos - model.geom_size, axis=0)
    upper = np.max(data.geom_xpos + model.geom_size, axis=0)
    np.testing.assert_allclose(upper - lower, size, atol=1e-12)
    np.testing.assert_allclose(lower, [-size[0] / 2, -size[1] / 2, 0], atol=1e-12)


@pytest.mark.parametrize("environment,count", [("table_shelf", 12), ("warehouse", 13)])
def test_furnished_layout_preserves_mpd_robot_relative_positions(environment, count):
    model = create_environment(environment).compile()
    data = initialize_data(model)
    assert model.nq == 0
    assert model.ngeom == count
    mount = data.site("robot_mount").xpos
    np.testing.assert_allclose(mount, [0, 0, 0.8])
    np.testing.assert_allclose(data.geom("table/box").xpos - mount, [0.45, 0, -0.4])
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
    if environment == "warehouse":
        np.testing.assert_allclose(
            data.geom("small_shelf/box").xpos - mount,
            [0.4, -0.6, -0.27],
            atol=1e-12,
        )


@pytest.mark.parametrize("name", ROBOT_NAMES)
def test_composition_preserves_robot_frames_controls_and_single_floor(name):
    original, original_data = create_robot(name)
    model, data = create_robot(name, environment="warehouse")
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
    model = spec.compile()
    data = initialize_data(model)
    assert data.ncon == 0
    data.qpos[2] = 0.8275
    mujoco.mj_forward(model, data)
    contacted = {model.geom(int(index)).name for contact in data.contact for index in contact.geom}
    assert "probe" in contacted
    assert "large_shelf/shelf_1" in contacted


@pytest.mark.parametrize("robot,nq", [(None, 0), ("panda", 9)])
def test_environment_cli_runs_outside_project_directory(robot, nq, tmp_path):
    command = [
        sys.executable,
        "-m",
        "mujoco_lab",
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
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=True
    )
    output = json.loads(result.stdout)
    assert len(output["qpos"]) == nq
    assert output["warnings"] == 0


def test_cli_rejects_environment_with_raw_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "simulate",
            "--environment",
            "warehouse",
            "--model",
            "unused.xml",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--environment cannot be combined with --model" in result.stderr


def test_unknown_environment_is_rejected():
    with pytest.raises(ValueError, match="Unknown environment 'missing'"):
        create_environment("missing")
