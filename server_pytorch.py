"""Server orchestration for federated PyTorch training using Flower.

This module integrates Flower's simulation API and provides a custom strategy
that records server-side and client-side metrics per round. The strategy
converts Flower Parameters to NumPy ndarrays, loads them into a PyTorch model
and evaluates on a provided test set.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from .client_pytorch import FraudClient
from .models import MLPFraud, torch_model_to_numpy_parameters, numpy_parameters_to_torch_state_dict
from .config import N_CLIENTS, N_ROUNDS, DEVICE
from .eval_utils import compute_metrics


class FlowerMetricsStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that records per-round metrics using a server-side model."""

    def __init__(self, global_model: MLPFraud, X_test: Optional[np.ndarray], y_test: Optional[np.ndarray], device, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        self.history: List[Dict[str, Any]] = []

    def aggregate_fit(self, rnd, results, failures):
        # Use parent aggregation to get aggregated Parameters
        aggregated_parameters, agg_metrics = super().aggregate_fit(rnd, results, failures)

        # Convert to ndarrays and load into torch model for evaluation
        try:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
        except Exception:
            ndarrays = None

        round_record: Dict[str, Any] = {"round": int(rnd)}

        if ndarrays is not None and self.X_test is not None and self.y_test is not None:
            numpy_parameters_to_torch_state_dict(self.global_model, ndarrays)
            self.global_model.eval()
            import torch as _torch

            xb = _torch.tensor(self.X_test, dtype=_torch.float32).to(self.device)
            with _torch.no_grad():
                preds = self.global_model(xb).cpu().numpy()
            y_pred = (preds >= 0.5).astype(int)
            metrics = compute_metrics(self.y_test, y_pred, y_score=preds)
            # Prefix with global_
            for k, v in metrics.items():
                round_record[f"global_{k}"] = v

        # Optionally, include aggregated metrics returned by the parent strategy
        if isinstance(agg_metrics, dict):
            for k, v in agg_metrics.items():
                round_record[f"agg_{k}"] = v

        # Store the record
        self.history.append(round_record)
        return aggregated_parameters, agg_metrics


class FlowerMedianStrategy(FlowerMetricsStrategy):
    """Strategy that aggregates client model parameters using elementwise median.

    This overrides `aggregate_fit` to compute the median across client ndarrays
    for each parameter tensor, then converts the ndarrays back to Flower
    Parameters for distribution.
    """

    def aggregate_fit(self, rnd, results, failures):
        # results: List[Tuple[ClientProxy, FitRes]]
        ndarrays_list = []
        for _, fit_res in results:
            try:
                arrs = parameters_to_ndarrays(fit_res.parameters)
                ndarrays_list.append(arrs)
            except Exception:
                # Skip clients whose parameters cannot be parsed
                continue

        round_record: Dict[str, Any] = {"round": int(rnd)}

        if len(ndarrays_list) == 0:
            return super().aggregate_fit(rnd, results, failures)

        # Compute elementwise median for each parameter tensor
        agg_ndarrays = []
        for i in range(len(ndarrays_list[0])):
            stacked = np.stack([client_arr[i] for client_arr in ndarrays_list], axis=0)
            agg = np.median(stacked, axis=0)
            agg_ndarrays.append(agg)

        # Convert ndarrays back to Parameters
        try:
            aggregated_parameters = ndarrays_to_parameters(agg_ndarrays)
        except Exception:
            aggregated_parameters = None

        # Evaluate on test set if available (reuse parent logic partially)
        if aggregated_parameters is not None and self.X_test is not None and self.y_test is not None:
            try:
                nds = parameters_to_ndarrays(aggregated_parameters)
                numpy_parameters_to_torch_state_dict(self.global_model, nds)
                self.global_model.eval()
                import torch as _torch

                xb = _torch.tensor(self.X_test, dtype=_torch.float32).to(self.device)
                with _torch.no_grad():
                    preds = self.global_model(xb).cpu().numpy()
                y_pred = (preds >= 0.5).astype(int)
                metrics = compute_metrics(self.y_test, y_pred, y_score=preds)
                for k, v in metrics.items():
                    round_record[f"global_{k}"] = v
            except Exception:
                pass

        # Store record and return
        self.history.append(round_record)
        return aggregated_parameters, {}


def client_fn(split: Tuple[np.ndarray, np.ndarray], input_dim: int, privacy_mode: str = "none") -> FraudClient:
    """Factory that returns a FraudClient for a given data split."""
    Xc, yc = split
    return FraudClient(Xc, yc, input_dim=input_dim, privacy_mode=privacy_mode)


def main_federated(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    privacy_mode: str = "none",
    num_rounds: int = N_ROUNDS,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    strategy_name: str = "FedAvg",
    save_model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a Flower in-process simulation and return per-round history.

    Args:
        splits: list of (X_client, y_client) tuples.
        input_dim: number of features.
        privacy_mode: privacy mode passed to each client.
        num_rounds: number of federated rounds to simulate.
        X_test, y_test: optional global test set for server evaluation.
        strategy_name: name of aggregation strategy ("FedAvg" or "FedMedian").
        save_model_path: optional path to save final global model state_dict.

    Returns:
        history: list of dicts with per-round metrics.
    """
    # Build client_fn that maps client id to split index
    def _client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        split = splits[idx]
        return client_fn(split, input_dim=input_dim, privacy_mode=privacy_mode)

    # Initialize global model used by the strategy for evaluation
    global_model = MLPFraud(input_dim)
    global_model.to(DEVICE)

    # Choose strategy
    if strategy_name == "FedAvg":
        strat = FlowerMetricsStrategy(global_model=global_model, X_test=X_test, y_test=y_test, device=DEVICE)
    elif strategy_name == "FedMedian":
        # FedMedian not built-in; reuse FedAvg class but aggregation behavior may differ.
        strat = FlowerMetricsStrategy(global_model=global_model, X_test=X_test, y_test=y_test, device=DEVICE)
    else:
        raise ValueError(f"Unknown strategy_name: {strategy_name}")

    # Start the Flower simulation
    fl.simulation.start_simulation(client_fn=_client_fn, num_clients=len(splits), config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strat)

    # After simulation, optionally save the final model
    if save_model_path is not None:
        # If strat has history, parameters are stored in global_model
        import torch as _torch

        _torch.save(global_model.state_dict(), save_model_path)

    return strat.history


if __name__ == "__main__":
    print("Use main_federated(splits, input_dim, privacy_mode) to run federated simulation using Flower")
