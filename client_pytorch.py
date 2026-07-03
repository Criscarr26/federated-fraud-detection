"""Flower NumPyClient wrapping a local PyTorch trainer for fraud detection.

The client receives local data (X_client, y_client) as NumPy arrays.
It trains for `LOCAL_EPOCHS` epochs (default 1 per federated round) and
returns updated parameters as a list of NumPy arrays.
"""
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl

from .models import MLPFraud, torch_model_to_numpy_parameters, numpy_parameters_to_torch_state_dict
from .config import DEVICE, LOCAL_EPOCHS, LEARNING_RATE
from .privacy import apply_privacy
from .eval_utils import compute_metrics


class FraudClient(fl.client.NumPyClient):
    """Flower client for PyTorch MLP.

    Args:
        X_client: Local features, numpy array.
        y_client: Local labels, numpy array.
        input_dim: Number of features.
        privacy_mode: mode string passed to apply_privacy.
    """

    def __init__(self, X_client: np.ndarray, y_client: np.ndarray, input_dim: int, privacy_mode: str = "none") -> None:
        self.X = X_client.astype(np.float32)
        self.y = y_client.astype(np.float32)
        self.model = MLPFraud(input_dim)
        self.model.to(DEVICE)
        self.privacy_mode = privacy_mode

    def get_parameters(self) -> List[np.ndarray]:
        """Return current model parameters as NumPy arrays."""
        return torch_model_to_numpy_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set local model parameters from NumPy arrays."""
        numpy_parameters_to_torch_state_dict(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Optional[dict]) -> Tuple[List[np.ndarray], int, dict]:
        """Train local model for LOCAL_EPOCHS and return updated parameters.

        This method applies privacy (clipping/noising) to the returned parameters
        as configured via self.privacy_mode.
        """
        # Load parameters
        self.set_parameters(parameters)

        dataset = TensorDataset(torch.tensor(self.X), torch.tensor(self.y))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Get parameters and compute local metrics
        params = self.get_parameters()
        # Compute local predictions for reporting
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(self.X).to(DEVICE)
            preds = self.model(xb).cpu().numpy()
        y_pred = (preds >= 0.5).astype(int)
        local_metrics = compute_metrics(self.y, y_pred, y_score=preds)

        # Privacy hyperparams; in a real experiment these should be configurable
        priv_params = apply_privacy(params, mode=self.privacy_mode, max_norm=1.0, sigma=0.01)
        # Return: parameters, number of examples, optional metrics
        return priv_params, len(self.X), {"local_metrics": local_metrics}

    def evaluate(self, parameters: List[np.ndarray], config: Optional[dict]) -> Tuple[float, int, dict]:
        """Optional: Evaluate current model on local data and return loss and metrics."""
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(self.X).to(DEVICE)
            preds = self.model(xb).cpu().numpy()
        # Return dummy loss (0) and number of examples
        return float(0.0), len(self.X), {"accuracy": float((preds >= 0.5).mean())}
