"""Closed-loop evaluation with small policies and deterministic stand-in environments."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config import BCTrainConfig, EvaluationConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.evaluation import PolicyEvaluator
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.policies.gaussian import GaussianPolicy


class TinyEnv:
    physics_steps_per_action = 1
    task = None

    def __init__(self, *, low=-10.0, high=10.0, finish=3, success=False, dt=0.002):
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = SimpleNamespace(
            shape=(1,), low=np.array([low]), high=np.array([high]), dtype=np.float32
        )
        self.simulator = SimpleNamespace(dt=dt, data=SimpleNamespace(time=0.0))
        self.finish = finish
        self.wins = success
        self.actions = []
        self.seeds = []
        self.steps = 0

    @property
    def action_dt(self):
        return self.simulator.dt * self.physics_steps_per_action

    def reset(self, *, seed):
        self.steps = 0
        self.simulator.data.time = 0.0
        self.seeds.append(seed)
        return np.array([2.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(float(action[0]))
        self.steps += 1
        self.simulator.data.time += self.simulator.dt
        ended = self.steps == self.finish
        success = ended and self.wins
        return (
            np.array([2.0 + self.steps], dtype=np.float32),
            float(success),
            success,
            ended and not success,
            {"success": success, "termination_reason": "success" if success else "time_limit"},
        )


def checkpoint(tmp_path, *, policy_type="mse", chunk_size=2, execution_horizon=2):
    config = BCTrainConfig(
        robot="forte",
        policy_type=policy_type,
        hidden_dims=(),
        obs_horizon=2,
        chunk_size=chunk_size,
        execution_horizon=execution_horizon,
        action_execution_hz=500,
    )
    model = build_policy(
        policy_type, state_dim=2, action_dim=1, chunk_size=chunk_size, hidden_dims=()
    )
    normalizer = Normalizer(
        state_mean=np.array([1.0], dtype=np.float32),
        state_std=np.array([2.0], dtype=np.float32),
        action_mean=np.array([10.0], dtype=np.float32),
        action_std=np.array([2.0], dtype=np.float32),
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        normalizer,
        config,
        optimizer_step=7,
        dataset_metadata={"replay": {"physics_steps_per_action": 1}},
    )
    return load_checkpoint(path)


def evaluate(env, model, normalizer, metadata, *, num_episodes=1, seed=50, max_steps=5):
    return PolicyEvaluator(
        env,
        EvaluationConfig(
            num_episodes=num_episodes,
            seed=seed,
            max_seconds=max_steps * env.action_dt,
            video_episodes=0,
        ),
        torch.device("cpu"),
    ).evaluate(model, normalizer, metadata, flow_num_steps=3)


def test_mse_history_padding_chunk_alignment_raw_goals_and_success(tmp_path):
    model, stats, metadata = checkpoint(tmp_path, chunk_size=3)
    policy_inputs = []
    handle = model.net[0].register_forward_pre_hook(
        lambda module, args: policy_inputs.append(args[0].detach().cpu().numpy().copy())
    )
    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([[0.0, 1.0]] * 3))
        model.net[0].bias.zero_()
    env = TinyEnv(low=0, high=11, finish=3, success=True)
    rows, summary = evaluate(env, model, stats, metadata, num_episodes=2)
    handle.remove()
    # First padded history [2,2] normalizes to [0.5,0.5], hence 10+2*0.5=11.
    # At the next chunk boundary history [3,4] gives the current-frame action 13,
    # forwarded unchanged; a fresh chunk is not sampled after the success step.
    assert env.actions == [11, 11, 13] * 2
    np.testing.assert_allclose(np.concatenate(policy_inputs), [[0.5, 0.5], [1.0, 1.5]] * 2)
    assert [row["steps"] for row in rows] == [3, 3]
    assert all(row["success"] for row in rows)
    assert all(row["termination_reason"] == "success" for row in rows)
    assert rows[0]["sim_seconds"] == pytest.approx(0.006)
    assert summary["mean_success_sim_seconds"] == pytest.approx(0.006)


def test_gaussian_evaluation_uses_mean_and_preserves_training_rng():
    model = GaussianPolicy(2, 1, 1, hidden_dims=())
    model.mean_head.load_state_dict({"weight": torch.zeros(1, 2), "bias": torch.tensor([0.25])})
    with torch.no_grad():
        model.log_std.fill_(2.0)
    normalizer = Normalizer(*(np.array([value], dtype=np.float32) for value in (0, 1, 0, 1)))
    metadata = {"architecture": {"obs_horizon": 2, "execution_horizon": 1}}
    env = TinyEnv(finish=3, success=True)
    rng = torch.random.get_rng_state().clone()
    rows, summary = evaluate(env, model, normalizer, metadata)
    assert env.actions == [0.25] * 3
    assert rows[0]["success"] and summary["success_rate"] == 1
    assert model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("max_steps,failure_reason", [(1, None), (5, "failed_stack")])
def test_terminated_failure_is_not_a_timeout(tmp_path, max_steps, failure_reason):
    model, stats, metadata = checkpoint(tmp_path)

    class FailedEnv(TinyEnv):
        def step(self, action):
            obs, reward, _, _, _ = super().step(action)
            return (
                obs,
                reward,
                True,
                False,
                {
                    "success": np.bool_(False),
                    "termination_reason": failure_reason,
                },
            )

    rows, summary = evaluate(FailedEnv(), model, stats, metadata, max_steps=max_steps)
    assert rows[0]["termination_reason"] == (failure_reason or "terminated")
    assert not rows[0]["success"]
    assert summary["timeouts"] == 0


def test_flow_seed_reproducibility_and_rng_and_mode_restoration(tmp_path):
    model, stats, metadata = checkpoint(tmp_path, policy_type="flow")
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].bias.zero_()
    model.train()
    rng_before = torch.random.get_rng_state().clone()
    env1, env2 = TinyEnv(), TinyEnv()
    rows1, _ = evaluate(env1, model, stats, metadata, num_episodes=2)
    rows2, _ = evaluate(env2, model, stats, metadata, num_episodes=2)
    assert rows1 == rows2
    assert env1.actions == env2.actions
    assert env1.seeds == env2.seeds == [50, 51]
    assert [row["env_seed"] for row in rows1] == [50, 51]
    assert all("policy_seed" not in row for row in rows1)
    assert model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    different = TinyEnv()
    evaluate(different, model, stats, metadata, seed=71)
    assert different.actions != env1.actions[: len(different.actions)]


def test_evaluator_restores_model_and_rng_when_step_raises(tmp_path):
    model, normalizer, metadata = checkpoint(tmp_path, policy_type="flow")

    class FailingEnv(TinyEnv):
        def step(self, action):
            raise RuntimeError("step failed")

    evaluator = PolicyEvaluator(
        FailingEnv(), EvaluationConfig(video_episodes=0), torch.device("cpu")
    )
    model.train()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="step failed"):
        evaluator.evaluate(model, normalizer, metadata, flow_num_steps=3)
    assert model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize(("action_dt", "expected_steps"), [(0.01, 3), (0.02, 2)])
def test_seconds_limit_uses_environment_cadence_and_stops_inside_chunk(
    tmp_path, action_dt, expected_steps
):
    model, stats, metadata = checkpoint(tmp_path, chunk_size=4, execution_horizon=4)
    env = TinyEnv(dt=action_dt, finish=100)
    rows, summary = PolicyEvaluator(
        env,
        EvaluationConfig(num_episodes=1, max_seconds=0.025, video_episodes=0),
        torch.device("cpu"),
    ).evaluate(model, stats, metadata, flow_num_steps=1)
    assert rows[0]["steps"] == expected_steps
    assert rows[0]["sim_seconds"] == pytest.approx(expected_steps * action_dt)
    assert summary["timeouts"] == 1


@pytest.mark.parametrize("max_seconds", [0.0, float("nan")])
def test_evaluator_rejects_invalid_durations(max_seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        PolicyEvaluator(TinyEnv(), EvaluationConfig(max_seconds=max_seconds), torch.device("cpu"))
