"""Privacy utilities: clipping and adding noise to model parameters."""
from typing import List
import numpy as np


def clip_parameters(parameters: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    """Clip the list of parameter arrays to have total l2-norm <= max_norm."""
    # compute total norm
    total = 0.0
    for p in parameters:
        total += np.sum(p.astype(np.float64) ** 2)
    norm = float(np.sqrt(total))
    if norm <= max_norm:
        return parameters
    scale = max_norm / (norm + 1e-12)
    return [p * scale for p in parameters]


def add_noise(parameters: List[np.ndarray], sigma: float) -> List[np.ndarray]:
    """Add Gaussian noise N(0, sigma^2) elementwise to each parameter array."""
    noisy = []
    for p in parameters:
        noise = np.random.normal(loc=0.0, scale=sigma, size=p.shape).astype(p.dtype)
        noisy.append(p + noise)
    return noisy


def apply_privacy(parameters: List[np.ndarray], mode: str = "none", max_norm: float = 1.0, sigma: float = 0.01) -> List[np.ndarray]:
    """Apply privacy transformations.

    Modes:
    - "none": no change
    - "clipping": clip to max_norm
    - "noising": add gaussian noise with sigma
    - "clipping+noising": clip then add noise
    """
    if mode == "none":
        return parameters
    if mode == "clipping":
        return clip_parameters(parameters, max_norm)
    if mode == "noising":
        return add_noise(parameters, sigma)
    if mode == "clipping+noising":
        clipped = clip_parameters(parameters, max_norm)
        return add_noise(clipped, sigma)
    raise ValueError(f"Unknown privacy mode: {mode}")
