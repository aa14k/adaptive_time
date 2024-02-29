from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import os
import numpy as np

import unicodedata
import re


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
    """ Argmax operation on the last axis---randomly break ties

    Cannot actually handle ndim > 1 matrices where the last
    dimension is not singleton.
    
    - x (np.ndarray): values

    """
    a, _ = argmax_with_probs(x, calc_action=True, calc_probs=False)
    return a


def argmax_with_probs(
        x: np.ndarray, *, calc_action, calc_probs
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """ Argmax operation on the last axis---randomly break ties

    Cannot actually handle ndim > 1 matrices where the last
    dimension is not singleton.
    
    - x (np.ndarray): values

    """
    if x.ndim > 1:
        raise ValueError("Cannot handle ndim > 1 matrices?")
    max_val = np.max(x, axis=-1)
    idxes = np.where(x == max_val)[0]
    action = np.random.choice(idxes) if calc_action else None

    if calc_probs:
        probs = np.zeros(x.shape[-1])
        probs[idxes] = 1 / len(idxes)
    else:
        probs = None

    return action, probs


def eps_greedy_policy_probs(epsilon, qs):
    num_actions = qs.shape[-1]
    eps_probs = epsilon * np.ones(num_actions) / num_actions
    _, greedy_probs = argmax_with_probs(
        qs, calc_action=False, calc_probs=True)
    total_probs = (1 - epsilon) * greedy_probs + eps_probs
    return total_probs


def v_from_eps_greedy_q(epsilon, qs):
    action_probs = eps_greedy_policy_probs(epsilon, qs)
    return np.dot(action_probs, qs)


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


def find_root_directory(path: str) -> str:
    """Finds the subpath to the root dir of the project in `path`."""
    where = path.find("adaptive_time")
    if where == -1:
        raise ValueError(
            f"Cannot find the root directory of the project in '{path}'.")
    return path[:where + len("adaptive_time")]


def set_directory_in_project(
        rel_path: Optional[str] = None, create_dirs: bool = False) -> str:
    """Changes the working dir to rel_path from the root of the project."""
    path = find_root_directory(os.getcwd())
    if rel_path is not None:
        path = os.path.join(path, rel_path)
    if create_dirs:
        os.makedirs(path, exist_ok=True)
    os.chdir(path)
    new_dir = os.getcwd()
    print("Changed working directory to", new_dir)
    return new_dir


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    This version taken from: https://stackoverflow.com/a/295466.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def get_abs_path(my_path: str) -> str:
    if os.path.isabs(my_path):
        return my_path
    else:
        return os.path.join(os.getcwd(), my_path)
