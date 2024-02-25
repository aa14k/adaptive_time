"""Implements the Monte Carlo control method used in the notebooks."""

import numpy as np

from adaptive_time import samplers
from adaptive_time import utils


def phi_sa(phi_x, a, prev_phi_sa=None):
    """Form the (state, action) feature, potentially reusing memory.
    
    - phi_x: the state feature
    - a: the action
    - prev_phi_sa: the previous state,action feature, which can be
      reused to avoid memory allocation.

    Returns the feature as a (2, d) array. Use a flat copy.
    """
    if prev_phi_sa is not None:
        prev_phi_sa.fill(0)
        phi_sa = prev_phi_sa
    else:
        phi_sa = np.zeros((2, phi_x.size))
    phi_sa[a] = phi_x
    return phi_sa


def ols_monte_carlo(
        trajectory, sampler: samplers.Sampler2, tqdm,
        phi, weights, targets, features, x0, gamma = 0.999):
    
    N = len(trajectory)
    pivots = sampler.pivots(trajectory)
    print(f"Using {len(pivots)}/{N} samples.")

    # TODO: optimize this.
    all_returns = utils.discounted_returns(trajectory, gamma)

    G = 0
    x_sa = np.zeros((2, phi.num_parameters))
    returns_a0 = []  # from x0 (the initial state), action 0
    returns_a1 = []  # from x0 (the initial state), action 1
    for t in tqdm(range(N-1,-1,-1)):
        state, action, reward, _ = trajectory[t]
        if t in pivots:
            prev_G = G
            G = gamma * G + reward
            if G != all_returns[t]:
                print(f"Error in G: {G} != {all_returns[t]}")
                assert False
            x = phi.get_fourier_feature(state)
            # Record empirical returns.
            if np.linalg.norm(x-x0) < 0.00001:
                if action == 0:
                    returns_a0.append(G)
                    returns_a1.append(-0)
                elif action == 1:
                    returns_a1.append(G)
                    returns_a0.append(-0)

            x_sa = phi_sa(x, action, x_sa)
            x_sa_flat = x_sa.flatten()

            features += np.outer(x_sa_flat, x_sa_flat)
            targets += G * x_sa_flat
        else:
            prev_G = G
            G = gamma * G + reward

        try:
            weights = np.linalg.solve(features, targets)
        except np.linalg.LinAlgError:
            print("Singular matrix in OLS. Using previous weights.")
    return weights, targets, features, (np.mean(returns_a0), np.mean(returns_a1))
