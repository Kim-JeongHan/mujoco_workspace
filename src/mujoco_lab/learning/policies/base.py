# 추가 구현 방향
# 공통 입력 state=(B, N_s), action_chunk=(B, C, N_a)와 scalar loss 계약을 유지한다.
# 입력 차원과 device/dtype 규약을 명확히 하고 개별 policy에서도 같은 형태를 반환한다.
# 행동 정규화는 공통 전처리에서 한 번 적용하는 방식으로 맞춘다.
# PPO의 log_prob·entropy 기능은 지원 가능한 policy의 별도 인터페이스로 둔다.
# 학습 중 MSEPolicy가 sample_actions를 호출하므로 공통 메서드에 무조건 no_grad를 붙이지 않는다.

from __future__ import annotations

import abc

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for policies that predict chunks of actions."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        """Initialize the dimensions shared by action-chunking policies.

        Args:
            state_dim: Number of state features, N_s.
            action_dim: Number of action features per step, N_a.
            chunk_size: Number of actions in each predicted chunk, C.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute the scalar training loss for a batch.

        Args:
            state (batch, dim): B N_s
            action_chunk (batch, chunk, dim): B C N_a

        Returns:
            torch.Tensor: Scalar loss with shape ().
        """

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate an action chunk from a batch of states.

        Args:
            state (batch, dim): B N_s
            num_steps: Number of sampling steps for iterative policies.

        Returns:
            torch.Tensor: Sampled action chunk with shape (B, C, N_a).
        """
