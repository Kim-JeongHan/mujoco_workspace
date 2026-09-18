import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mujoco_lab import create_robot

EXAMPLE = Path(__file__).parents[1] / "examples" / "manipulator.py"


@pytest.fixture
def headless_env():
    env = {**os.environ, "MUJOCO_GL": "disable"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    return env


@pytest.mark.parametrize("name,nq", [("ur20", 6), ("ur30", 6), ("panda", 9), ("forte", 7)])
def test_cli_selects_robot_outside_project_directory(name, nq, tmp_path, headless_env):
    result = subprocess.run(
        [sys.executable, "-m", "mujoco_lab", "simulate", "--robot", name, "--steps", "100"],
        cwd=tmp_path,
        env=headless_env,
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert len(output["qpos"]) == nq
    assert output["simulated_seconds"] == pytest.approx(0.2)
    assert output["warnings"] == 0


@pytest.mark.parametrize("name", ["ur20", "ur30", "panda", "forte"])
@pytest.mark.parametrize("environment", ["empty", "warehouse"])
def test_manipulator_example_runs_headless(name, environment, tmp_path, headless_env):
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--robot",
            name,
            "--environment",
            environment,
            "--headless",
            "--steps",
            "100",
        ],
        cwd=tmp_path,
        env=headless_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"{name}: simulated 0.200 seconds" in result.stdout
    assert f"environment = {environment}" in result.stdout


def test_cli_rejects_conflicting_model_selection(tmp_path, headless_env):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "simulate",
            "--robot",
            "panda",
            "--model",
            "unused.xml",
        ],
        cwd=tmp_path,
        env=headless_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_factory_rejects_unknown_robot():
    with pytest.raises(ValueError, match="Unknown robot 'ur5'. Available robots:"):
        create_robot("ur5")
