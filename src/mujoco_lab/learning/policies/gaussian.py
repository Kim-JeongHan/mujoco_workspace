# 추가 구현 방향
# 현재 누락된 sample_actions를 구현해야 BasePolicy의 추상 메서드가 충족된다.
# forward의 mean.view(...) 뒤 쉼표 때문에 mean이 tuple이 되므로 해당 부분을 수정해야 한다.
# 분포 샘플과 평균 행동을 선택할 수 있게 하고 log_std의 허용 범위를 정한다.
# PPO용으로 저장된 행동의 log_prob와 entropy를 재계산하는 메서드를 제공한다.
# 행동 차원 축을 합산해 샘플별 값을 반환하고 초기 PPO는 chunk_size=1로 연결한다.
# 행동을 tanh 등으로 변환한다면 확률 계산도 그 변환에 맞춰 처리한다.
# BC 평균 가중치·정규화를 이어받고 탐색 표준편차 초기화는 명시적으로 설정한다.

from __future__ import annotations

import torch
from torch import nn

from mujoco_lab.learning.policies.base import BasePolicy


class GaussianPolicy(BasePolicy):
    """Predict Gaussian distributions over chunks of actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        """Initialize the state-conditioned Gaussian policy.

        Args:
            state_dim: Number of state features, N_s.
            action_dim: Number of action features per step, N_a.
            chunk_size: Number of actions in each predicted chunk, C.
            hidden_dims: Width of each hidden MLP layer.
        """
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.net = nn.Sequential(*layers)

        self.mean_head = nn.Linear(
            input_dim,
            action_dim * chunk_size,
        )

        self.log_std = nn.Parameter(torch.zeros(chunk_size, action_dim))

    def forward(self, state):
        """Return the Gaussian distribution conditioned on a batch of states.

        Args:
            state (batch, dim): B N_s

        Returns:
            torch.distributions.Normal: Distribution with mean and standard
                deviation broadcastable to shape (B, C, N_a).
        """
        batch = state.shape[0]

        h = self.net(state)

        mean = self.mean_head(h)
        mean = (mean.view(batch, self.chunk_size, self.action_dim),)

        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)

    def compute_loss(self, state, action_chunk):
        """Compute the negative log likelihood of expert action chunks.

        Args:
            state (batch, dim): B N_s
            action_chunk (batch, chunk, dim): B C N_a

        Returns:
            torch.Tensor: Mean negative log likelihood with shape ().
        """
        dist: torch.distributions.Normal = self(state)

        log_prob = dist.log_prob(action_chunk)

        return -log_prob.sum(dim=(-1, -2)).mean()
