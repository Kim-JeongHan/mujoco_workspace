"""Stateful on-policy and off-policy actor-critic agents.

Agents own policy and critic networks, optimizer state, and learning iterations.
On-policy trainers supply rollouts with fixed advantage and return targets::

    agent = DAPGAgent(policy, critic, config)
    agent.demonstrations = demonstrations
    metrics = agent.update(rollout)

DDPGAgent instead updates from a normalized ReplayBatch sampled with optional HER.
PPOAgent currently initializes learning state; PPO policy updates remain pending.
"""

from mujoco_lab.learning.agents.base import Agent, OnPolicyAgent
from mujoco_lab.learning.agents.dapg import DAPGAgent, Demonstrations
from mujoco_lab.learning.agents.ddpg import DDPGAgent
from mujoco_lab.learning.agents.npg import NPGAgent
from mujoco_lab.learning.agents.ppo import PPOAgent

__all__ = [
    "Agent",
    "OnPolicyAgent",
    "NPGAgent",
    "DAPGAgent",
    "PPOAgent",
    "DDPGAgent",
    "Demonstrations",
]
