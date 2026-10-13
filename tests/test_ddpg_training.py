"""DDPG gradients, target semantics, actual MuJoCo training, and exact resume."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.agents import DDPGAgent
from mujoco_lab.learning.config.ddpg import DDPGConfig
from mujoco_lab.learning.config.evaluation import EvaluationConfig
from mujoco_lab.learning.envs.factory import create_env
from mujoco_lab.learning.evaluate_ddpg import run as evaluate_ddpg
from mujoco_lab.learning.models.critic import QCritic
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.rl_checkpoint import (
    load_ddpg_checkpoint,
    restore_rng,
    save_ddpg_checkpoint,
)
from mujoco_lab.learning.rollout.replay_buffer import ReplayBatch
from mujoco_lab.learning.trainers import train_ddpg


def batch(states, actions, rewards, terminated, truncated):
    return ReplayBatch(
        states,
        actions,
        rewards,
        states + 1,
        terminated,
        truncated,
        torch.zeros(len(states), dtype=torch.bool),
    )


def test_bellman_targets_bootstrap_truncation_and_actor_updates_through_q():
    policy = MSEPolicy(1, 1, 1, hidden_dims=(), bounded=True)
    critic = QCritic(1, 1, hidden_dims=())
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        weight, bias = tuple(critic.parameters())
        weight.copy_(torch.tensor([[0.5, 1.0]]))
        bias.zero_()
    config = DDPGConfig(gamma=0.9, tau=0.25, device="cpu")
    agent = DDPGAgent(policy, critic, config)
    with torch.no_grad():
        weight, bias = tuple(agent.target_critic.parameters())
        weight.zero_()
        bias.fill_(3)
    original_targets = [parameter.clone() for parameter in agent.target_policy.parameters()]
    minibatch = batch(
        torch.ones(3, 1),
        torch.zeros(3, 1),
        -torch.ones(3),
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
    )
    metrics = agent.update(minibatch)
    targets = torch.tensor([-1.0, 1.7, 1.7])
    assert metrics["critic/mean_target"] == pytest.approx(targets.mean().item())
    assert metrics["critic/loss"] == pytest.approx(((targets - 0.5) ** 2).mean().item())
    assert policy(torch.ones(1, 1)).item() > 0
    assert policy(torch.full((1, 1), 1e4)).abs().max() <= 1
    for previous, target, online in zip(
        original_targets, agent.target_policy.parameters(), policy.parameters(), strict=True
    ):
        torch.testing.assert_close(target, previous * 0.75 + online * 0.25)
        assert not target.requires_grad and target.grad is None
    assert agent.actor_optimizer.state and agent.critic_optimizer.state


def test_ddpg_learns_a_known_continuous_action_optimum():
    torch.manual_seed(7)
    agent = DDPGAgent(
        MSEPolicy(1, 1, 1, hidden_dims=(32, 32), bounded=True),
        QCritic(1, 1, hidden_dims=(32, 32)),
        DDPGConfig(device="cpu", actor_learning_rate=0.003, critic_learning_rate=0.003),
    )
    before = abs(agent.policy(torch.ones(1, 1)).item() - 0.5)
    for _ in range(400):
        actions = torch.rand(128, 1) * 2 - 1
        minibatch = batch(
            torch.ones(128, 1),
            actions,
            -((actions[:, 0] - 0.5) ** 2),
            torch.ones(128, dtype=torch.bool),
            torch.zeros(128, dtype=torch.bool),
        )
        agent.update(minibatch)
    error = abs(agent.policy(torch.ones(1, 1)).item() - 0.5)
    assert error < 0.1 and error < before


def small_config(tmp_path, **kwargs):
    return DDPGConfig(
        robot="panda",
        device="cpu",
        hidden_dims=(8,),
        critic_hidden_dims=(8,),
        physics_steps_per_action=2,
        obs_horizon=2,
        max_seconds=0.016,
        total_steps=8,
        batch_size=8,
        warmup_steps=0,
        updates_per_episode=2,
        noise_std=0.5,
        random_action_probability=0.5,
        replay_capacity=16,
        output_dir=tmp_path / "logs",
        eval_interval_episodes=1,
        checkpoint_interval_episodes=1,
        evaluation=EvaluationConfig(num_episodes=1, max_seconds=0.012, video_episodes=0),
        **kwargs,
    )


def test_checkpoint_replays_the_next_physical_episode_and_update_exactly(tmp_path):
    config = small_config(tmp_path)
    env = create_env(train_ddpg.scene_config(config), max_steps=4, terminate_on_success=False)
    try:
        torch.manual_seed(3)
        np.random.seed(3)
        state = train_ddpg.initialize_training(config, env)
        train_ddpg.train_episode(state, env, max_steps=4)
        path = tmp_path / "ddpg.pt"
        save_ddpg_checkpoint(path, state)
        expected = train_ddpg.train_episode(state, env, max_steps=4)
        restored = load_ddpg_checkpoint(path)
        assert restored.rng is not None
        restore_rng(restored.rng)
        actual = train_ddpg.train_episode(restored, env, max_steps=4)
        assert actual == pytest.approx(expected, abs=0, rel=0)
        assert restored.agent.iteration == state.agent.iteration == 4
        assert restored.moments.count == state.moments.count
        np.testing.assert_array_equal(restored.normalizer.state_std, state.normalizer.state_std)
        for model, other in (
            (state.agent.policy, restored.agent.policy),
            (state.agent.critic, restored.agent.critic),
            (state.agent.target_policy, restored.agent.target_policy),
            (state.agent.target_critic, restored.agent.target_critic),
        ):
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, other.state_dict()[key], rtol=0, atol=0)
        for episode, other in zip(state.replay.episodes, restored.replay.episodes, strict=True):
            np.testing.assert_array_equal(episode.observations, other.observations)
            np.testing.assert_array_equal(episode.actions, other.actions)
            assert np.all(np.abs(episode.actions) <= 1 + 1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("scene", ["cube_stack", "book_insertion"])
def test_real_training_evaluation_and_resume_preserve_the_action_budget(tmp_path, scene):
    config = small_config(tmp_path, scene=scene)
    run_dir = train_ddpg.run(replace(config, total_steps=7))
    checkpoint = run_dir / "checkpoint.pt"
    state = load_ddpg_checkpoint(checkpoint)
    assert (state.env_steps, state.episodes, state.agent.iteration) == (7, 2, 4)
    assert len(state.replay) == 7
    assert len(list((run_dir / "eval").glob("*/summary.json"))) == 2
    resumed_dir = train_ddpg.run(
        DDPGConfig(resume=checkpoint, device="cpu", total_steps=9, output_dir=config.output_dir)
    )
    resumed = load_ddpg_checkpoint(resumed_dir / "checkpoint.pt")
    assert (resumed.env_steps, resumed.episodes, resumed.agent.iteration) == (9, 3, 6)
    assert resumed.agent.config.scene == scene
    metrics = evaluate_ddpg(
        resumed_dir / "checkpoint.pt", episodes=1, video_episodes=0, output_dir=config.output_dir
    )
    assert metrics["eval/attempted"] == 1 and metrics["train/env_steps"] == 9
    assert 0 <= metrics["eval/goal_success_rate"] <= 1
    assert all(np.isfinite(value) for value in metrics.values())
