from abc import ABC, abstractclassmethod
from typing import Any, List, Tuple

import numpy as np

from adaptive_time import utils
from adaptive_time import value_est


class AproxIntegrator(ABC):
    """AproxIntegrator return approximate integrals and the poitns they used."""

    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        """Returns the approx integral and the array of indices used."""
        pass


def _integrate_with_quadrature(rewards, quadrature_fn, tolerance):
    N = len(rewards)

    rewards_with_idxs = np.array([rewards, np.arange(N)])
    # for idx, transition_or_reward in enumerate(rewards_with_idxs):
    #     if isinstance(transition_or_reward, (list, tuple)):
    #         rewards_with_idxs[0, idx] = transition_or_reward[2]
    #     else:
    #         rewards_with_idxs[0, idx] = transition_or_reward
    #     rewards_with_idxs[1, idx] = idx
    used_idxes = {}
    integral = quadrature_fn(rewards_with_idxs, tolerance, used_idxes)
    pivots = list(sorted(used_idxes.keys()))
    return integral, np.array(pivots, dtype=np.int32)


class AdaptiveQuadratureIntegrator(AproxIntegrator):
    """Identifies pivots using quadrature methods, access to total sum."""

    def __init__(self, tolerance: float) -> None:
        super().__init__()
        self._tolerance = tolerance
    
    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        return _integrate_with_quadrature(
            rewards, approx_integrate, self._tolerance)


class UniformIntegrator(AproxIntegrator):
    """Returns uniformly spaced pivots."""
    def __init__(self, spacing: int) -> None:
        super().__init__()
        self._spacing = spacing

    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        N = len(rewards)
        rewards_with_idxs = np.array([rewards, np.arange(N)])
        # TODO: this does not necessarily include the tail. Should we
        # for to include it?
        spacing_pivots = np.arange(0, N, self._spacing)
        print("num_values: ", N)
        print("spacing_pivots: ", spacing_pivots)
        integral = 0
        used_idxes = {}
        for idx_into_pivots in range(len(spacing_pivots)):
            # print("first idx: ", spacing_pivots[idx_into_pivots])
            idx_range = spacing_pivots[idx_into_pivots:idx_into_pivots+2].copy()
            # print("idx_range: ", idx_range.shape, idx_range)
            if len(idx_range) > 1:
                idx_range[1] -= 1
                if idx_range[0] == idx_range[1]:
                    idx_range = idx_range[0:1]  # A single item, as an array.
            # print("idx_range: ", idx_range.shape, idx_range)
            integral_part = _trapezoid_approx(
                rewards_with_idxs[:, idx_range],
                used_idxes)
            print("  ->  part idx", idx_range, f": {integral_part}")
            integral += integral_part
        pivots = list(sorted(used_idxes.keys()))
        return integral, np.array(pivots, dtype=np.int32)


def approx_integrate(xs, tol, idxes):
    """Approximately integrate using the adaptive quadrature method.
    
    Approximates the integral of all `xs[0]`s, using the trapezoidal rule.
    Critically, only a subset of the indices are used to approximate the integral.
    These are placed in the `idxes` dictionary.

    NOTE: this implementation does not access true integrals towards a stopping
    criterion, but instead iteratively checks if a one-iteration better
    approximation changes the integral by more than `tol`.

    Args:
    - xs (2D np.ndarray): the input data, contains the function values
        and their corresponding indices.
    - tol (float): the tolerance for the approximation.
    - idxes (Dict): an empty dictionary to store the indices in that we used.

    Returns:
        The approximate integral. (Note, idxes is modified in place.)
    """
    # TODO: the current implementation is not very efficient, basically
    # everything is re-calculated twice.
    N = len(xs[0])
    Q_est = _trapezoid_approx(xs, idxes)
    # Now find a one better approximation, and check if we're good enough.
    c = int(np.floor(N / 2))
    Q_better = (
        _trapezoid_approx(xs[:,:c], idxes)
        + _trapezoid_approx(xs[:,c:], idxes))
    # print()
    # print(xs)
    # print("   -->   Q_est, Q_better, tol", Q_est, Q_better, tol)
    # print()
    if np.abs(Q_est - Q_better) > tol:
        Q_est = (
            approx_integrate(xs[:,:c], tol / 2, idxes)
            + approx_integrate(xs[:,c:], tol / 2, idxes))
    return Q_est


def _trapezoid_approx(xs, idxes):
    """Use the trapezoid method on the first and last points of xs.
    
    Args:
    - xs ((2, N) array): the input data, contains the function values
        and their corresponding indices.
    - idxes (Dict): a dictionary to store the indices in that we used.
        Updated in place.
        
    Returns: the approximate integral.
    """
    N = len(xs[0])

    idxes[int(xs[1,0])] = 1
    idxes[int(xs[1,-1])] = 1

    if N > 2:
        Q = N * (xs[0,0] + xs[0,-1]) / 2        
    else:
        Q = sum(xs[0])

    # print(f"- trap  Q: {Q};   from\n{xs}")
    return Q
