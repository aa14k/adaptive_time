import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax operation on the last axis
    - x (np.ndarray): logits

    """
    z = x - np.max(x, axis=-1)
    return np.exp(z) / np.sum(np.exp(z), axis=-1)
