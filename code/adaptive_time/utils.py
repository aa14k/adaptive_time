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


def argmax(x: np.ndarray) -> np.ndarray:
    """
    Argmax operation on the last axis---randomly break ties
    - x (np.ndarray): values

    """
    max_val = np.max(x, axis=-1)
    idxes = np.where(x == max_val)[0]
    return np.random.choice(idxes)


def discounted_returns(traj, gamma):
    """Return all discounted returns for a trajectory."""
    disc_returns = []
    for i, st in enumerate(reversed(traj)):
        if i == 0:
            disc_returns.append(st[2])
        else:
            disc_returns.append(st[2] + gamma * disc_returns[-1])

    return list(reversed(disc_returns))


def total_returns(traj):
    """Return all discounted returns for a trajectory."""
    rewards = np.array([st[2] for st in traj])
    return np.flip(np.cumsum(np.flip(rewards)))


def approx_integrate(xs, tol, idxes):
    """Approximately integrate using quadrature method.
    
    Approximates the integral of all `xs[0]`s, using the trapezoidal rule.
    Critically, only a subset of the indices are used to approximate the integral.
    These are placed in the `idxes` dictionary.

    NOTE: this implementation assumes we can access the full integral, and
    use that to decide if our approximation is good enough already.

    Args:
    - xs (2D np.ndarray): the input data, contains the function values
        and their corresponding indices.
    - tol (float): the tolerance for the approximation.
    - idxes (Dict): an empty dictionary to store the indices in that we used.

    Returns:
        The approximate integral.
    """
    N = len(xs[0])
    if N > 2:
        Q = N * (xs[0,0] + xs[0,-1]) / 2
        idxes[int(xs[1,0])] = 1
        idxes[int(xs[1,-1])] = 1
    else:
        idxes[int(xs[1,0])] = 1
        idxes[int(xs[1,-1])] = 1
        return sum(xs[0])
    truth = sum(xs[0])
    if np.abs(Q - truth) > tol:
        c = int(np.floor(N / 2))
        Q = (
            approx_integrate(xs[:,:c], tol / 2, idxes)
            + approx_integrate(xs[:,c:], tol / 2, idxes))
    return Q
    