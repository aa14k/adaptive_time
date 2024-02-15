from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from q_functions import QFunction


def generate_traj(
    env: Any,
    q_function: Any,
) -> Tuple[Dict[str, Any], int]:
    """
    Generate a trajectory from the environment using the most fine-grain timescale
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from

    """
    curr_obs = env.reset()
    done = False
    obss = [curr_obs]
    acts = []
    rews = []
    horizon = 0
    while not done:
        curr_act = q_function.greedy_action(curr_obs)
        reward, curr_obs, horizon, done = env.step(curr_act)
        obss.append(curr_obs)
        acts.append(curr_act)
        rews.append(reward)

    return {
        "obss": np.array(obss),
        "acts": np.array(acts),
        "rews": np.array(rews),
    }, horizon


def observe_discrete_traj(
    cont_traj: Dict[str, Any],
    observe_times: List[int],
) -> Dict[str, Any]:
    """
    Observes the discretized trajectory based on provided sampling time indices
    - cont_traj (Dict[str, Any]): The "continuous-time" trajectory
    - observe_times (List[int]): The sampling time indices

    """
    return {
        key: value[observe_times] for key, value in cont_traj.items()
    }


def mc_policy_iteration(
    env: Any,
    q_function: QFunction,
    observation_sampler: Any,
    config: SimpleNamespace
):
    """
    Runs Monte Carlo Policy Iteration
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from
    - observation_sampler (Any): The sampler that indicates which time to observe the state
    - config (SimpleNamespace): The configuration of the learning algorithm

    config.budget: total number of samples we can observe
    config.num_trajs_per_update: the number of trajectories per policy iteration step

    """
    
    budget = config.budget
    num_trajs_per_update = config.num_trajs_per_update

    sample_i = 0
    while sample_i < budget:
        traj_i = 0
        observe_times = observation_sampler.sample_time()

        cont_trajs = []
        disc_trajs = []
        while traj_i < num_trajs_per_update:
            # Observe "continuous-time" trajectory
            cont_traj, horizon = generate_traj(env, q_function)
            cont_trajs.append(
                cont_traj
            )
            
            # Discretize trajectory
            disc_trajs.append(
                observe_discrete_traj(cont_traj, observe_times)
            )
            sample_i += len(observe_times)
            traj_i += 1

            # Check if we have broken the budget
            if sample_i >= budget:
                break

        q_function.update(disc_trajs)
