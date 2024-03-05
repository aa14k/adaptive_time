
import numpy as np


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
    """Approximately integrate using the trapezoid method."""
    N = len(xs[0])

    idxes[int(xs[1,0])] = 1
    idxes[int(xs[1,-1])] = 1

    if N > 2:
        Q = N * (xs[0,0] + xs[0,-1]) / 2        
    else:
        Q = sum(xs[0])

    # print(f"- trap  Q: {Q};   from\n{xs}")
    return Q

