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
