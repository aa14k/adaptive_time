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
            rewards, adaptive_approx_integrate, self._tolerance)


class AdaptiveQuadratureIntegratorOld(AproxIntegrator):
    """Identifies pivots using quadrature methods, access to total sum."""

    def __init__(self, tolerance: float) -> None:
        super().__init__()
        self._tolerance = tolerance
    
    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        return _integrate_with_quadrature(
            rewards, approx_integrate_old, self._tolerance)


class UniformlySpacedIntegrator(AproxIntegrator):
    """Returns uniformly spaced pivots."""
    def __init__(self, spacing: int) -> None:
        super().__init__()
        self._spacing = spacing

    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        N = len(rewards)
        rewards_with_idxs = np.array([rewards, np.arange(N)])

        spacing_pivots = np.arange(0, N, self._spacing)
        # print("N", N)
        # print("spacing_pivots", spacing_pivots)
        # We ensure we include the last data point.
        if spacing_pivots[-1] != N-1:
            spacing_pivots = np.append(spacing_pivots, N-1)
        # print("spacing_pivots", spacing_pivots)
    
        integral = 0
        used_idxes = {}
        for idx_into_pivots in range(len(spacing_pivots)-1):
            # We are guaranteeed that consecutive pivots are not
            # the same, so we can use them as indices.
            idx_low = spacing_pivots[idx_into_pivots]
            idx_high = spacing_pivots[idx_into_pivots+1]
            # We want to include idx_high, but ranges in python don't:
            # print(f" * part idx[{idx_low}:{idx_high+1}]:")
            integral_part = _trapezoid_approx(
                rewards_with_idxs[:, idx_low:idx_high+1],
                used_idxes)
            # Subtract the last point, as it will be added in the next part.            
            last_point = rewards_with_idxs[0, idx_high]
            # print(f"      -> {integral_part} - last={last_point}")
            integral += integral_part - last_point
        # The very last point we'll have to add back:
        integral += last_point
        # print(f"  -> readd final point: {last_point}")
        pivots = list(sorted(used_idxes.keys()))
        return integral, np.array(pivots, dtype=np.int32)


class UniformlySpacedIntegratorOld(AproxIntegrator):
    """Returns uniformly spaced pivots."""
    def __init__(self, spacing: int) -> None:
        super().__init__()
        self._spacing = spacing

    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        N = len(rewards)
        rewards_with_idxs = np.array([rewards, np.arange(N)])

        spacing_pivots = np.arange(0, N, self._spacing)
        # print("N", N)
        # print("spacing_pivots", spacing_pivots)
        # We ensure we include the last data point.
        if spacing_pivots[-1] != N:
            spacing_pivots = np.append(spacing_pivots, N)
        # print("spacing_pivots", spacing_pivots)
    
        integral = 0
        used_idxes = {}
        for idx_into_pivots in range(len(spacing_pivots)-1):
            idx_low = spacing_pivots[idx_into_pivots]
            idx_high = spacing_pivots[idx_into_pivots+1]
            # print(f" * part idx[{idx_low}:{idx_high}]:")
            integral_part = _trapezoid_approx(
                rewards_with_idxs[:, idx_low:idx_high],
                used_idxes)
            # print(f"      -> {integral_part}")
            integral += integral_part
        pivots = list(sorted(used_idxes.keys()))
        return integral, np.array(pivots, dtype=np.int32)


def adaptive_approx_integrate(xs, tol, idxes):
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
    if N == 0:
        return 0
    elif N <= 2:
        # Works for N=1 as well.
        idxes[int(xs[1,0])] = 1
        idxes[int(xs[1,-1])] = 1
        return sum(xs[0])
    
    # We now have a sequence of 3 points; we can split down the center.
    # Use the trapezoid method on the first and last points, then find
    # a (potentially) better approximation by splitting. Check if our
    # approximation improved by more than tol, if no, we're good to stop.
    # Otherwise recursively call this function on the two halves.
    Q_est = _trapezoid_approx(xs, idxes)

    c = int(np.floor(N / 2))
    Q_better = (
        _trapezoid_approx(xs[:,:c+1], idxes)  # Include c.
        + _trapezoid_approx(xs[:,c:], idxes)  # Include c.
        - xs[0,c]  # Remove the double-counted c.
    )  # c was already added to the idxes in the _trapzoid_approx calls.
    # print()
    # print(xs)
    # print("   -->   Q_est, Q_better, tol", Q_est, Q_better, tol)
    # print()
    if np.abs(Q_est - Q_better) > tol:
        Q_est = (
            adaptive_approx_integrate(xs[:,:c+1], tol / 2, idxes)
            + adaptive_approx_integrate(xs[:,c:], tol / 2, idxes)
            - xs[0,c]  # Remove the double-counted c.
        )
        
    return Q_est


def approx_integrate_old(xs, tol, idxes):
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
    if N == 0:
        return 0
    elif N == 1:
       idxes[int(xs[1,0])] = 1
       return xs[0][0]
    
    # Otherwise we have at least two points and we can split the integral.
    # TODO: this could be optimized, for 2 points we don't need to split.
    Q_est = _trapezoid_approx(xs, idxes)

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
            approx_integrate_old(xs[:,:c], tol / 2, idxes)
            + approx_integrate_old(xs[:,c:], tol / 2, idxes))
        
    return Q_est


def _trapezoid_approx(xs, idxes):
    """Use the trapezoid method on the first and last points of xs.
    
    Args:
    - xs ((2, N) array): the input data, contains the function values
        and their corresponding indices. Should have length at least 1.
    - idxes (Dict): a dictionary to store the indices in that we used.
        Updated in place.
        
    Returns: the approximate integral.
    """
    N = len(xs[0])

    if N == 0:
        return 0
    
    # Update the idxes dictionary.
    idxes[int(xs[1,0])] = 1
    idxes[int(xs[1,-1])] = 1

    if N > 2:
        Q = N * (xs[0,0] + xs[0,-1]) / 2.0
    else:
        Q = sum(xs[0])

    # print(f"- trap  Q: {Q};   from\n{xs}")
    return Q
