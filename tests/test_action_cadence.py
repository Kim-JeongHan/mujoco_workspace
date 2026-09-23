"""Action cadence keeps physical control fresh and episode steps distinct."""

from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.control import create_controller
from mujoco_lab.learning.datasets.replay import capture_frame
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout import collect_episode
from mujoco_lab.tasks import CubeStackTask


def make_env(repeat):
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    robot = simulator.robots["forte"]
    robot.change_controller(create_controller("pd", robot, frame="grasp"))
    task = CubeStackTask(simulator, 2)
    return CubeStackEnv(task, physics_steps_per_action=repeat, max_steps=2), robot


@pytest.mark.parametrize("repeat", [1, 5])
def test_action_holds_target_with_fresh_control_each_physics_tick(monkeypatch, repeat):
    env, robot = make_env(repeat)
    env.reset(seed=7)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(released_stable_stack=False))
    action = np.clip(np.zeros(env.action_space.shape), env.action_space.low, env.action_space.high)
    targets = []
    qpos_samples = []
    original_control = robot.control

    def control():
        targets.append(robot.target.position.copy())
        qpos_samples.append(robot.state.snapshot().qpos.copy())
        return original_control()

    monkeypatch.setattr(robot, "control", control)
    start_time = env.data.time
    obs, _, terminated, truncated, info = env.step(action)
    assert obs.shape == (54,)
    assert not terminated and not truncated
    assert env._steps == 1
    assert env.action_dt == pytest.approx(0.002 * repeat)
    assert env.data.time - start_time == pytest.approx(env.action_dt)
    assert info["physics_steps"] == repeat
    assert info["elapsed_dt"] == pytest.approx(env.action_dt)
    assert len(targets) == len(qpos_samples) == repeat
    for target in targets[1:]:
        np.testing.assert_array_equal(target, targets[0])
    if repeat == 5:
        assert not np.array_equal(qpos_samples[0], qpos_samples[-1])


def test_success_can_end_partial_action_and_reset_clears_terminal_state(monkeypatch):
    env, _ = make_env(5)
    env.reset(seed=7)
    calls = 0

    def status():
        nonlocal calls
        calls += 1
        return SimpleNamespace(released_stable_stack=calls == 2)

    monkeypatch.setattr(env.task, "status", status)
    action = np.clip(np.zeros(env.action_space.shape), env.action_space.low, env.action_space.high)
    _, reward, terminated, truncated, info = env.step(action)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["physics_steps"] == 2
    assert info["elapsed_dt"] == pytest.approx(0.004)
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(action)
    env.reset(seed=8)
    assert env._steps == 0
    assert not env._done


@pytest.mark.parametrize("repeat", [True, 0, -1, 1.5])
def test_action_repeat_requires_positive_integer(repeat):
    with pytest.raises(ValueError, match="physics_steps_per_action"):
        make_env(repeat)


def test_collector_records_actual_repeat_and_action_boundary_frames(monkeypatch):
    env, _ = make_env(5)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(released_stable_stack=False))

    class FixedExpert:
        failed = False

        def reset(self, obs, info):
            pass

        def act(self, obs):
            return np.clip(
                np.zeros(env.action_space.shape), env.action_space.low, env.action_space.high
            ).astype(np.float32)

    episode = collect_episode(
        env,
        FixedExpert(),
        max_steps=2,
        seed=7,
        record_frame=lambda: capture_frame(env.simulator),
        replay_metadata={"dt": env.simulator.dt},
    )
    assert episode.metadata["replay"]["physics_steps_per_action"] == 5
    assert episode.qpos.shape[0] == episode.states.shape[0] == len(episode) + 1 == 3
    np.testing.assert_allclose(np.diff(episode.frame_times), 0.01)
    with pytest.raises(ValueError, match="Replay action cadence differs"):
        collect_episode(
            env,
            FixedExpert(),
            max_steps=1,
            replay_metadata={"physics_steps_per_action": 1},
        )
