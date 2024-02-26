import random

from adaptive_time.environments.mountain_car import MountainCar
from adaptive_time.environments.corridor import Corridor
from adaptive_time.environments.cartpole import CartPoleEnv
from adaptive_time.environments.cartpole import CartPoleVectorEnv


def create_trajectories(env_ctor, num_trajectories, policy):
    """Returns a list of trajectories generated by policy.

    Each trajectory is a tuple of (states, rewards), where

    * states is the list of states visited by the policy before the set
        timeout (horizon_sec). The list is roughly length `horizon_sec / dt_sec`.
    * rewards is the list of rewards received by the policy at each state.
        This list is one shorter than states.
    """
    trajectories = []
    env = env_ctor()
    for _ in range(num_trajectories):
        rewards = []
        states = []

        state = env.reset()
        states.append(state)

        is_done = False
        while not is_done:
            reward, state, _, is_done = env.step(policy(state))
            rewards.append(reward)
            states.append(state)
        trajectories.append((states, rewards))

    return trajectories


def generate_trajectory(
        env, seed=None, policy=None, termination_prob=0.0, max_steps=None):
    """Generates a single trajectory from the environment using the given policy.

    Note that the trajectory is generated until the environment terminates, and
    truncations are not handled!
    
    Args:
        env: The environment to generate the trajectory from.
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
    observation, _ = env.reset(seed=seed)
    trajectory = []
    terminated = False
    steps = 0
    if policy is None:
        policy = lambda x: env.action_space.sample()
    while not terminated:
        steps += 1
        action = policy(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        trajectory.append([observation, action, reward, observation_])
        observation = observation_
        if random.random() < termination_prob:
            terminated = True

        if steps % 20_000 == 0:
            pass
            #print('Did 20_000 steps!', steps)
        
        if max_steps is not None and steps > max_steps:
            return trajectory, True
            #print('Max steps reached!', steps)

    return trajectory, False

