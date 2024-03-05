"""Implements the Monte Carlo control method used in the notebooks."""

import numpy as np

from adaptive_time import samplers
from adaptive_time import utils


# _TARGET_SCALAR = 1000.
_TARGET_SCALAR = 1.


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
        # 2 is the number of actions, which is fixed for now
        phi_sa = np.zeros((2, phi_x.size))
    phi_sa[a] = phi_x
    return phi_sa


def ols_monte_carlo(
        trajectory, sampler: samplers.Sampler2, tqdm,
        phi, weights, targets, features, x0, do_weighing, gamma, ep_num):
    """Processes a trajectory to update the weights using OLS Monte Carlo.
    
    Args:
    - trajectory: the trajectory to process
    - sampler: the sampler used to subsample from the trajectory
    - tqdm: pass a tqdm function to use for progress bars, or simply an identity function
    - phi: an instance of fourier feature function class
    - weights: UNUSED.
    - targets: previous targets, will be updated with data from the new trajectory
    - features: previous features, will be updated with data from the new trajectory
    - x0: the initial state (used only for reporting)
    - do_weighing: whether to weigh updates according to the associated length of the
        pivots they use.
    - gamma: the discount factor
    - scale: the scaling factor for normalizing features and targets
    - ep_num: the episode number (should count from 1)
    """

    N = len(trajectory)
    pivots = sampler.pivots(trajectory)
    print(f"Using {len(pivots)}/{N} samples.")

    # Could optimize the below by iterating only over pivots,
    # and using the discounted returns from `all_returns` directly.
    all_returns = utils.discounted_returns(trajectory, gamma)
    all_pivot_features = np.zeros((len(pivots), 2*phi.num_parameters))

    pivot_idx = -1  # Start with the last one, count down.
    prev_pivot = N-1
    G = 0

    # 2 is the number of actions.
    x_sa = np.zeros((2, phi.num_parameters))
    returns_a0 = []  # from x0 (the initial state), action 0
    returns_a1 = []  # from x0 (the initial state), action 1
    features_dt = np.zeros_like(features)
    targets_dt = np.zeros_like(targets)
    all_dts = []

    for t in tqdm(range(N-1,-1,-1)):
        state, action, reward, _ = trajectory[t]
        G = gamma * G + reward
        if t in pivots:
            # Process the pivot.
            if G != all_returns[t]:
                print(f"Error in G: {G} != {all_returns[t]}")
                assert False
            x = phi.get_fourier_feature(state)
            # Record empirical returns.
            if np.linalg.norm(x-x0) < 0.00001:
                if action == 0:
                    returns_a0.append(G)
                elif action == 1:
                    returns_a1.append(G)

            x_sa = phi_sa(x, action, x_sa)
            x_sa_flat = x_sa.flatten()
            all_pivot_features[pivot_idx] = x_sa_flat
            if t == N-1:
                dt = 1   # The weight for the last update.
            else:
                dt = prev_pivot - t
                prev_pivot = t
            if not do_weighing:
                dt = 1
            all_dts.append(dt)
            # # the scale is increasing over time, so we need to scale the features
            features_dt = (
                features_dt + dt * np.outer(x_sa_flat, x_sa_flat) / len(pivots))
            targets_dt = targets_dt + dt * G * x_sa_flat / len(pivots)
            # features_dt = features_dt + dt * np.outer(x_sa_flat, x_sa_flat)
            # targets_dt = targets_dt + dt * G * x_sa_flat
            pivot_idx -= 1

    # Update the average features and targets.
    features = features + (1./(ep_num+1.0)) * (features_dt - features)
    targets = targets + (1./(ep_num+1.0)) * (targets_dt/_TARGET_SCALAR - targets)

    # print("dt's used: ", all_dts)

    try:
        # weights = np.linalg.solve(features / scale, targets / scale)
        # (weights, _, rank, _) = np.linalg.lstsq(
        #     features / scale,
        #     targets / scale
        # )
        # weights = np.linalg.solve(features, targets)
        (weights, _, rank, _) = np.linalg.lstsq(
            features,
            targets
        )
        # print(np.min(x_sa_flat), np.max(x_sa_flat))
        # # print(x_sa_flat.shape, features.shape, targets.shape)
        # # print(weights.shape)
        print("rank:", rank)
        # print("feat: {} targ: {}".format(
        #     np.linalg.norm(features/scale, ord=1), np.linalg.norm(targets/scale, ord=1)))
        print("param-norm: {}".format(np.linalg.norm(weights)))
        # print("residual: {}".format(np.linalg(
        #         (features @ weights - targets) / scale), ord=1))
        print("residual: {}".format(np.linalg.norm(
                features @ weights - targets), ord=1))
        # print(np.sum((features / scale) @ weights - (targets / scale)))

        weights *= _TARGET_SCALAR
    except np.linalg.LinAlgError:
        print("Singular matrix in OLS. Using previous weights.")
    
    # Sanity check on the quality of the weights.
    _targets = np.array(all_returns)[pivots]
    errors = np.abs(all_pivot_features @ weights - _targets)
    mean_err_pivots = np.mean(errors)
    max_err_pivots = np.max(errors)
    print("Error in predictions (mean, max) -- on pivots:",
          mean_err_pivots, max_err_pivots)

    est_return_qs = (np.nanmean(returns_a0), np.nanmean(returns_a1))
    est_return_v = np.mean(returns_a0 + returns_a1)
    return (
        weights, targets, features,
        est_return_qs, est_return_v,
        len(pivots),
        mean_err_pivots, max_err_pivots
    )
