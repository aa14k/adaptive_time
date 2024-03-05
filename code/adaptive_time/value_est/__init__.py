
import numpy as np
import random


# def reset_randomness(seed, env):
#     """Resets all randomness, except for the environment's seed."""
#     random.seed(seed)
#     np.random.seed(seed)
#     # env.seed(seed)
#     env.action_space.seed(seed)    


def generate_trajectory(
        env, start_state=None, seed=None, policy=None,
        termination_prob=0.0, max_steps=None):
    """Generates a single trajectory from the environment using the given policy.

    NOTE: the policy gets the current timestep as well!

    Note that the trajectory is generated until the environment terminates, and
    truncations are not handled!
    
    Args:
        env: The environment to generate the trajectory from.
        start_state: The state to start the trajectory from.
        seed: The seed to use for the environment. Defaults to None.
        policy: A function that takes in a state and returns an action. Defaults
            to a random policy if not provided.
        termination_prob: The probability of terminating the trajectory at each
            step. Defaults to 0.0.
        max_steps: The maximum number of steps to take before terminating the
            interaction. Defaults to None.
    
    Returns:
        (traj, is_early_terminated), where
        * traj is list of transitions, where each transition is a tuple of
            `(state, action, reward, next_state)`.
        * is_early_terminated is a boolean indicating if the trajectory was
            terminated early due to `max_steps`.
    """
    observation, _ = env.reset(seed=seed, start_state=start_state)
    trajectory = []
    terminated = False
    steps = 0
    if policy is None:
        policy = lambda x: env.action_space.sample()
    while not terminated:
        action = policy(observation, steps)
        steps += 1
        observation_, reward, terminated, truncated, info = env.step(action)
        trajectory.append([observation, action, reward, observation_])
        observation = observation_
        if random.random() < termination_prob:
            terminated = True

        if steps % 20_000 == 0:
            print('Did 20_000 steps!', steps)
        
        if max_steps is not None and steps > max_steps:
            return trajectory, True
            #print('Max steps reached!', steps)

    return trajectory, False





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

