"""Model definitions and helper functions.

Includes a PyTorch MLP and helper functions to convert parameters between
PyTorch tensors and NumPy arrays for federated aggregation.
"""
from typing import List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn


class MLPFraud(nn.Module):
    """Simple MLP for fraud detection.

    Binary output with sigmoid activation.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def torch_model_to_numpy_parameters(model: nn.Module) -> List[np.ndarray]:
    """Convert model parameters to a list of NumPy arrays (for Flower)."""
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def numpy_parameters_to_torch_state_dict(model: nn.Module, params: List[np.ndarray]) -> None:
    """Set model state_dict values from a list of NumPy arrays (in order)."""
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError("Parameter length mismatch when setting state dict")
    new_state = {}
    for k, p in zip(state_dict.keys(), params):
        new_state[k] = torch.tensor(p)
    model.load_state_dict(new_state)


def flatten_coef_intercept(coef: np.ndarray, intercept: np.ndarray) -> List[np.ndarray]:
    """Return flattened representation for aggregation (list of arrays)."""
    return [coef.copy(), intercept.copy()]


def coef_intercept_from_flat(flat: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Recover coef and intercept from flattened representation."""
    return flat[0], flat[1]
