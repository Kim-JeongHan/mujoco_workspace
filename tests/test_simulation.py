import os
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_environment


def test_custom_model_resolves_include_relative_to_its_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "robot"
    model_dir.mkdir()
    (model_dir / "scene.xml").write_text('<mujoco><include file="body.xml"/></mujoco>')
    (model_dir / "body.xml").write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    monkeypatch.chdir(tmp_path)
    scene = mujoco.MjSpec.from_file(str((model_dir / "scene.xml").resolve()))
    sim = Simulator(scene)
    initial_height = float(sim.data.qpos[2])
    sim.run_steps(50)

    assert sim.data.qpos[2] < initial_height
    assert sim.robots == {}


def test_scene_blueprint_can_be_reused_without_accumulating_robot_attachments():
    scene = create_environment("empty")
    first = Simulator(scene, robots=[RobotSpec("arm", "forte")])
    second = Simulator(scene, robots=[RobotSpec("arm", "forte")])

    assert scene.body("arm/base_link") is None
    assert first.model.nbody == second.model.nbody
    np.testing.assert_array_equal(first.data.qpos, second.data.qpos)


def test_home_state_is_captured_after_forward_and_fully_restored():
    scene = mujoco.MjSpec.from_string("""<mujoco>
      <worldbody>
        <body name="jointed"><joint name="hinge"/><geom size="0.1"/></body>
        <body name="marker" mocap="true"><geom size="0.01"/></body>
      </worldbody>
      <actuator><general joint="hinge" dyntype="filter" dynprm="0.1"/></actuator>
      <keyframe><key name="home" time="2" qpos="0.4" qvel="0.2" act="0.3"
        ctrl="0.5" mpos="1 2 3" mquat="1 0 0 0"/></keyframe>
    </mujoco>""")
    sim = Simulator(scene)
    expected = {
        name: getattr(sim.data, name).copy()
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "xpos")
    }
    assert sim.data.time == 2
    sim.data.time = 9
    for name in expected:
        getattr(sim.data, name)[:] = 0

    sim.reset()

    assert sim.data.time == 2
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(sim.data, name), value)


def test_model_example_simulates_an_external_model_without_robot_binding(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    example = Path(__file__).parents[1] / "examples" / "model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--model",
            str(path),
            "--headless",
            "--steps",
            "10",
        ],
        cwd=tmp_path,
        env={**os.environ, "MUJOCO_GL": "disable"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"{path}: simulated 0.020 seconds" in result.stdout


@pytest.mark.parametrize("controller", ["pd", "osc"])
def test_cli_uses_default_forte_for_feedback_control(controller, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--environment",
            "empty",
            "--controller",
            controller,
            "--steps",
            "3",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


def test_installed_cli_runs_without_display_from_another_directory(tmp_path):
    env = {**os.environ, "MUJOCO_GL": "disable"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--robot",
            "panda",
            "--steps",
            "100",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == result.stderr == ""


def test_cli_render_logs_saved_path_to_standard_logging_stream(tmp_path):
    output = tmp_path / "scene.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "render",
            "--environment",
            "empty",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "MUJOCO_GL": "egl"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == ""
    assert re.fullmatch(
        rf"\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}} \[INFO\] {re.escape(str(output))}\n",
        result.stderr,
    )
    assert output.stat().st_size > 1000


def test_cli_rejects_negative_step_count(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mujoco_lab", "--command", "simulate", "--steps", "-1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must be zero or greater" in result.stderr


def test_cli_defaults_to_forte_in_empty_environment(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mujoco_lab", "--command", "simulate", "--steps", "3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


def test_cli_rejects_removed_model_option(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--model",
            "scene.xml",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Unrecognized options: --model" in result.stderr
