import os
import subprocess
import sys
from pathlib import Path

import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment

EXAMPLE = Path(__file__).parents[1] / "examples" / "manipulator.py"


@pytest.fixture
def headless_env():
    env = {**os.environ, "MUJOCO_GL": "disable"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    return env


@pytest.mark.parametrize("name", ["panda", "forte"])
def test_cli_selects_robot_outside_project_directory(name, tmp_path, headless_env):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--robot",
            name,
            "--steps",
            "100",
        ],
        cwd=tmp_path,
        env=headless_env,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("name", ["panda", "forte"])
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


def test_factory_rejects_unknown_robot():
    with pytest.raises(KeyError, match="ur5"):
        Simulator(create_environment("empty"), robots=[RobotSpec("arm", "ur5")])
