from collections.abc import Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Sequence[int],
    activation_fn: type[torch.nn.Module] = torch.nn.ReLU,
) -> torch.nn.Sequential:
    """
    Build a multi-layer perceptron (MLP) model.

    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output features.
        hidden_layers (Sequence[int]): The number of units in each hidden layer.
        activation_fn (type[torch.nn.Module]): The activation class to instantiate between layers.

    Returns:
        torch.nn.Sequential: The constructed MLP model.
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_fn())
        prev_dim = hidden_dim

    layers.append(torch.nn.Linear(prev_dim, output_dim))

    return torch.nn.Sequential(*layers)
