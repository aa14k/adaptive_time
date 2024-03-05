
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

        if steps % 25_000 == 0:
            print('Did 25_000 steps!', steps)
        
        if max_steps is not None and steps > max_steps:
            return trajectory, True
            #print('Max steps reached!', steps)

    return trajectory, False


