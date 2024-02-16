from types import SimpleNamespace
from typing import Dict

import numpy as np


def parse_dict(d: Dict) -> SimpleNamespace:
    """
    Parse dictionary into a namespace.
    Reference: https://stackoverflow.com/questions/66208077/how-to-convert-a-nested-python-dictionary-into-a-simple-namespace

    :param d: the dictionary
    :type d: Dict
    :return: the namespace version of the dictionary's content
    :rtype: SimpleNamespace

    """
    x = SimpleNamespace()
    _ = [
        setattr(x, k, parse_dict(v)) if isinstance(v, dict) else setattr(x, k, v)
        for k, v in d.items()
    ]
    return x


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax operation on the last axis
    - x (np.ndarray): logits

    """
    z = x - np.max(x, axis=-1)
    return np.exp(z) / np.sum(np.exp(z), axis=-1)
