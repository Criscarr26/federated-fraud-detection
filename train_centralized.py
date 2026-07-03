"""Centralized training for baseline MLP fraud detection using PyTorch."""
from typing import Tuple
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from .models import MLPFraud
from .config import CENTRAL_EPOCHS, LEARNING_RATE, DEVICE


def train_centralized(X: np.ndarray, y: np.ndarray, scaler, save_dir: str = "./") -> Tuple[nn.Module, dict]:
    """Train an MLP centrally and return the trained model and metrics.

    Args:
        X: np.ndarray features (num_samples, num_features)
        y: np.ndarray labels (num_samples,)
        scaler: fitted scaler object to save
        save_dir: directory to save model and scaler

    Returns:
        model, metrics dict
    """
    os.makedirs(save_dir, exist_ok=True)
    input_dim = X.shape[1]
    model = MLPFraud(input_dim)
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(CENTRAL_EPOCHS):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{CENTRAL_EPOCHS}, loss={np.mean(epoch_losses):.4f}")

    # Evaluate on full training data
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    y_pred = (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, preds)),
    }
    # Save model and scaler
    model_path = os.path.join(save_dir, "central_mlp.pt")
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}, scaler saved to {scaler_path}")
    return model, metrics


if __name__ == "__main__":
    print("This module provides train_centralized(X, y, scaler, save_dir)")
