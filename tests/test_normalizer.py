"""Large float32 datasets retain accurate normalization statistics."""

import numpy as np

from mujoco_lab.learning.datasets.normalizer import Normalizer


def test_large_dataset_constant_and_varying_features_have_correct_statistics():
    states = np.full((1_000_000, 2), [0.82, 0.074], dtype=np.float32)
    states[::2, 1] = 0.05
    actions = np.full((1_000_000, 2), [0.42, 0.074], dtype=np.float32)
    actions[::2, 0] = 0.46
    states.flags.writeable = actions.flags.writeable = False
    low_state, high_state = float(states[0, 1]), float(states[1, 1])
    high_action, low_action = float(actions[0, 0]), float(actions[1, 0])

    normalizer = Normalizer.from_data(states, actions)

    np.testing.assert_allclose(normalizer.state_mean, [states[0, 0], (low_state + high_state) / 2])
    np.testing.assert_allclose(normalizer.state_std, [1e-6, (high_state - low_state) / 2])
    np.testing.assert_allclose(
        normalizer.action_mean, [(low_action + high_action) / 2, actions[0, 1]]
    )
    np.testing.assert_allclose(normalizer.action_std, [(high_action - low_action) / 2, 1e-6])
    normalized_states = normalizer.normalize_state(states[:2])
    normalized_actions = normalizer.normalize_action(actions[:2])
    np.testing.assert_allclose(normalized_states, [[0, -1], [0, 1]], atol=2e-6)
    np.testing.assert_allclose(normalized_actions, [[1, 0], [-1, 0]], atol=2e-6)
    assert normalized_states.dtype == normalized_actions.dtype == np.float32
    np.testing.assert_allclose(normalizer.denormalize_state(normalized_states), states[:2])
    np.testing.assert_allclose(normalizer.denormalize_action(normalized_actions), actions[:2])
