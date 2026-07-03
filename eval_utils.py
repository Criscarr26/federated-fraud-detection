"""Evaluation utilities: metrics, plotting and CSV saving."""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict[str, float]:
    """Compute common binary classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions (0/1).
        y_score: Optional probability scores for AUC.

    Returns:
        Dict with accuracy, precision, recall, f1, auc (if y_score provided)
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")
    return metrics


def plot_metrics_per_round(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot metrics stored per round.

    Args:
        history: Dict where keys are metric names and values are lists per round.
        save_path: File path to save the figure.
    """
    plt.figure(figsize=(8, 5))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Save a list of result dicts to CSV.

    Each dict should be flat (no nested objects) and represent a single experiment run.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
