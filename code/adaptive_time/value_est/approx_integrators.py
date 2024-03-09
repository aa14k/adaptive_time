from abc import ABC, abstractclassmethod
from typing import Any, List, Tuple

import numpy as np
import functools

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


class AdaptiveQuadratureIntegratorNew(AproxIntegrator):
    """TODO describe."""

    def __init__(self, tolerance: float) -> None:
        super().__init__()
        raise NotImplementedError("This is not bug-free yet!!!")
        self._tolerance = tolerance
    
    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        return _integrate_with_quadrature(
            rewards, adaptive_top_level, self._tolerance)


class AdaptiveQuadratureIntegrator(AproxIntegrator):
    """Identifies pivots using quadrature methods, access to total sum."""

    def __init__(self, tolerance: float, print_debug: bool = False) -> None:
        super().__init__()
        self._tolerance = tolerance
        self.print_debug = print_debug
        self._integrate_fn = functools.partial(
            adaptive_approx_integrate, print_debug=self.print_debug)
    
    def integrate(self, rewards) -> Tuple[float, np.ndarray]:
        return _integrate_with_quadrature(
            rewards, self._integrate_fn, self._tolerance)


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


def adaptive_approx_integrate(xs, tol, idxes, print_debug=False):
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
        if print_debug:
            print()
            print(f"  approx integrate; len={N}; base case for {xs[0]}")
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
    outside_tol = np.abs(Q_est - Q_better) > tol

    if print_debug:
        print()
        print(f"  approx integrate; len={N}, "
              f"s={xs[0,0]}, m={xs[0,c]}, e={xs[0,-1]}")
        print(f"   -->   Q_est={Q_est}, Q_better={Q_better},"
              f" error={np.abs(Q_est - Q_better)} "
              f"> tol={tol} ? => Recurse? {outside_tol}")

    if outside_tol:
        Q_better = (
            adaptive_approx_integrate(xs[:,:c+1], tol / 2, idxes, print_debug)
            + adaptive_approx_integrate(xs[:,c:], tol / 2, idxes, print_debug)
            - xs[0,c]  # Remove the double-counted c.
        )

    return Q_better


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


def simpsons_rule(ys,a,b, used_idxes, print_debug=False):
    if print_debug:
        print("simpsons_rule: ", a, b)
    # NOTE: b is included in the sum!!
    h = (b - a) / 3
    if h <= 1:
        for i in range(a,b+1):
            used_idxes[i] = 1
        return sum(ys[a:b+1])
    def f(x):
        return ys[int(x)]
    used_idxes[f(a)] = 1
    used_idxes[f(a+h)] = 1
    used_idxes[f(a+2*h)] = 1
    used_idxes[f(b)] = 1
    if print_debug:
        print("looking at: ", f(a), f(a+h), f(a+2*h), f(b))
    return 3 * h / 8 * (f(a) + 3 * f(a + h) + 3 * f(a + 2 * h) + f(b))

def Newton_Cotes(a,b):
    return (a + b) / 2

def adaptive(ys, a, b, tol, used_idxes, print_debug=False):
    if print_debug:
        print("adaptive: ", a, b)
    # NOTE: b is included in the sum!!
    # used_idxes[a] = 1
    # used_idxes[b] = 1
    if b - a <= 1:  # Base case: up to 2 elements.
        if print_debug:
            print("adaptive: base case")
        # TODO: mark a and b!
        return np.sum(ys[a:b+1])
    c = Newton_Cotes(a,b)
    c = int(np.floor(c))
    assert c != a and c != b, f"adaptive: c={c} a={a} b={b}"

    Sab = simpsons_rule(ys,a,b, used_idxes)
    Sac = simpsons_rule(ys,a,c, used_idxes)
    Scb = simpsons_rule(ys,c,b, used_idxes)
    Q = Sac + Scb - ys[c]   # Remove the double-counted c.
    if np.abs(Sab - Q) > tol:
        Q = (adaptive(ys, a,c,tol / 2, used_idxes)
             + adaptive(ys, c,b,tol / 2, used_idxes)
             - ys[c])  # Remove the double-counted c.
    return Q


def adaptive_top_level(rewards_with_idxs, tolerance, used_idxes, print_debug=False):
    rewards = rewards_with_idxs[0]  # we'll just pass the full array.
    return adaptive(rewards, 0, len(rewards)-1, tolerance, used_idxes, print_debug)


# def approx_int_new(xs, tol, idxes):