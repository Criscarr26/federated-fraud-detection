"""Configuration constants for federated_fraud project."""
from typing import List
import torch

N_CLIENTS: int = 3
N_ROUNDS: int = 5
LOCAL_EPOCHS: int = 1
BATCH_SIZE: int = 32
CENTRAL_EPOCHS: int = 10
LEARNING_RATE: float = 1e-3
PRIVACY_MODES: List[str] = ["none", "clipping", "noising", "clipping+noising"]
SEED: int = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
