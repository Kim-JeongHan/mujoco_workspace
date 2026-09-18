import json
import os
import subprocess
import sys

import mujoco
import pytest

from mujoco_lab import load_simulation


def test_custom_model_resolves_include_relative_to_its_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "robot"
    model_dir.mkdir()
    (model_dir / "scene.xml").write_text('<mujoco><include file="body.xml"/></mujoco>')
    (model_dir / "body.xml").write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    monkeypatch.chdir(tmp_path)
    model, data = load_simulation(model_dir / "scene.xml")
    initial_height = float(data.qpos[2])
    mujoco.mj_step(model, data, nstep=50)

    assert data.qpos[2] < initial_height


def test_installed_cli_runs_without_display_from_another_directory(tmp_path):
    env = {**os.environ, "MUJOCO_GL": "disable"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    result = subprocess.run(
        [sys.executable, "-m", "mujoco_lab", "simulate", "--robot", "panda", "--steps", "100"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)

    assert output["mujoco_version"] == "3.13.0"
    assert output["simulated_seconds"] == pytest.approx(0.2)
    assert output["warnings"] == 0


def test_cli_rejects_negative_step_count(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mujoco_lab", "simulate", "--steps", "-1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must be zero or greater" in result.stderr


def test_cli_requires_explicit_model_selection(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mujoco_lab", "simulate"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "select --robot, --model, or --environment" in result.stderr
