import os
import subprocess
import sys
from pathlib import Path

import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets.robot.robot import ControllerConfig, RobotConfig

EXAMPLE = Path(__file__).parents[1] / "examples" / "simulation.py"


@pytest.fixture
def headless_env():
    env = {**os.environ, "MUJOCO_GL": "disable"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    return env


@pytest.mark.parametrize("name", [None, "panda", "forte"])
@pytest.mark.parametrize("environment", ["empty", "warehouse"])
def test_simulation_example_runs_headless(name, environment, tmp_path, headless_env):
    command = [
        sys.executable,
        str(EXAMPLE),
        "--environment",
        environment,
        "--headless",
        "--steps",
        "100",
    ]
    if name is not None:
        command += ["--robot", name]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=headless_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        f"[INFO] environment={environment} robot={name or 'none'} simulated=0.200s" in result.stderr
    )


def test_factory_rejects_unknown_robot():
    with pytest.raises(KeyError, match="ur5"):
        Simulator(
            create_environment("empty"),
            robots=[RobotSpec("arm", "ur5", config=RobotConfig(ControllerConfig("none")))],
        )
