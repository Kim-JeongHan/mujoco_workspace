"""Rollout ownership, GAE integration, and minibatch alignment."""

import pytest
import torch

from mujoco_lab.learning.rollout.buffer import RolloutBuffer


def add_step(buffer, ids, *, terminated=None, truncated=None):
    """Encode transition IDs in each field to track shuffled alignment."""
    buffer.add(
        states=ids[:, None],
        actions=(ids + 10)[:, None, None].expand(-1, 2, 3),
        rewards=ids + 20,
        values=ids + 30,
        next_values=ids + 40,
        log_probs=ids + 50,
        terminated=torch.zeros_like(ids, dtype=torch.bool) if terminated is None else terminated,
        truncated=torch.zeros_like(ids, dtype=torch.bool) if truncated is None else truncated,
    )


def test_rollout_copies_inputs_and_retains_raw_actions_without_gradients():
    buffer = RolloutBuffer(1, 1, (2, 3), num_envs=2)
    ids = torch.tensor([0.0, 1.0], requires_grad=True)
    add_step(buffer, ids)
    with torch.no_grad():
        ids.fill_(-100)

    torch.testing.assert_close(buffer.states[0, :, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(buffer.actions[0, :, 0, 0], torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(buffer.log_probs[0], torch.tensor([50.0, 51.0]))
    assert len(buffer) == 1
    assert not buffer.states.requires_grad
    assert not buffer.actions.requires_grad
    assert not buffer.values.requires_grad
    assert not buffer.log_probs.requires_grad


def test_rollout_gae_preserves_terminal_and_truncation_semantics():
    buffer = RolloutBuffer(4, 1, (1,), num_envs=2, dtype=torch.float64)
    for step in range(2):
        buffer.add(
            states=torch.zeros(2, 1),
            actions=torch.zeros(2, 1),
            rewards=torch.tensor([step + 1.0, step + 1.0]),
            values=torch.zeros(2),
            next_values=torch.full((2,), 10.0 if step == 1 else 0.0),
            log_probs=torch.zeros(2),
            terminated=torch.tensor([step == 1, False]),
            truncated=torch.tensor([False, step == 1]),
        )
    buffer.compute_returns(gamma=0.9, gae_lambda=0.5)

    expected = torch.tensor([[1.9, 5.95], [2.0, 11.0]], dtype=torch.float64)
    torch.testing.assert_close(buffer.advantages[:2], expected)
    torch.testing.assert_close(buffer.returns[:2], expected)
    assert sum(len(batch.states) for batch in buffer.minibatches(3)) == 4


def test_minibatches_keep_fields_aligned_and_visit_each_transition_each_epoch():
    buffer = RolloutBuffer(3, 1, (2, 3), num_envs=2)
    for step in range(3):
        add_step(buffer, torch.tensor([2.0 * step, 2.0 * step + 1]))
    buffer.compute_returns(gamma=0.0)

    for _ in range(2):
        batches = list(buffer.minibatches(4))
        assert [len(batch.states) for batch in batches] == [4, 2]
        ids = torch.cat([batch.states[:, 0] for batch in batches])
        torch.testing.assert_close(ids.sort().values, torch.arange(6.0))
        for batch in batches:
            ids = batch.states[:, 0]
            assert batch.actions.shape == (len(ids), 2, 3)
            torch.testing.assert_close(batch.actions, (ids + 10)[:, None, None].expand(-1, 2, 3))
            torch.testing.assert_close(batch.old_values, ids + 30)
            torch.testing.assert_close(batch.old_log_probs, ids + 50)
            torch.testing.assert_close(batch.advantages, torch.full_like(ids, -10))
            torch.testing.assert_close(batch.returns, ids + 20)
            assert not batch.old_values.requires_grad
            assert not batch.old_log_probs.requires_grad


def test_reset_reuses_storage_and_excludes_previous_rollout():
    buffer = RolloutBuffer(3, 1, (2, 3))
    for step in range(3):
        add_step(buffer, torch.tensor([float(step)]))
    buffer.compute_returns()
    storage = buffer.states.data_ptr()

    buffer.reset()
    assert len(buffer) == 0
    assert buffer.states.data_ptr() == storage
    with pytest.raises(RuntimeError, match="compute returns"):
        list(buffer.minibatches(2))

    add_step(buffer, torch.tensor([100.0]), terminated=torch.tensor([True]))
    buffer.compute_returns()
    batches = list(buffer.minibatches(2))
    assert len(batches) == 1
    torch.testing.assert_close(batches[0].states, torch.tensor([[100.0]]))
    torch.testing.assert_close(batches[0].returns, torch.tensor([120.0]))


def test_adding_a_step_invalidates_targets_and_full_buffer_requires_reset():
    buffer = RolloutBuffer(2, 1, (2, 3))
    with pytest.raises(RuntimeError, match="empty rollout"):
        buffer.compute_returns()
    add_step(buffer, torch.tensor([0.0]))
    buffer.compute_returns()
    add_step(buffer, torch.tensor([1.0]))
    with pytest.raises(RuntimeError, match="compute returns"):
        list(buffer.minibatches(1))
    with pytest.raises(RuntimeError, match="full"):
        add_step(buffer, torch.tensor([2.0]))


def test_minibatch_size_must_be_positive():
    buffer = RolloutBuffer(1, 1, (2, 3))
    add_step(buffer, torch.tensor([0.0]))
    buffer.compute_returns()
    with pytest.raises(ValueError, match="positive"):
        list(buffer.minibatches(0))
