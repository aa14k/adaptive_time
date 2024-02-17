from pprint import pprint
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from adaptive_time.q_functions import QFunction


def generate_traj(
    env: Any,
    q_function: Any,
    max_horizon: int = None,
    action_repeat: int = 1,
    episode_i: int = 50,
) -> Tuple[Dict[str, Any], int]:
    """
    Generate a trajectory from the environment using the most fine-grain timescale
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from
    - max_horizon (int): Maximum horizon of the environment
    - use_action_repeat (bool): Whether to use action repeat, or get a chance to
        re-sample the action at each time step.
    - episode_i (int): The current episode number, used for environment reset

    """
    curr_obs = env.reset(episode_i)
    done = False
    obss = [curr_obs]
    acts = []
    rews = []
    horizon = 0
    while not done:
        curr_act = q_function.greedy_action(curr_obs)
        for _ in range(action_repeat + 1):
            reward, curr_obs, (_, horizon), done = env.step(curr_act)
            obss.append(curr_obs)
            acts.append(curr_act)
            rews.append(reward)
            if done:
                break

    if max_horizon is not None:
        padding_to_add = max_horizon - horizon - 1
        obss.extend([curr_obs] * padding_to_add)
        acts.extend([curr_act] * padding_to_add)
        rews.extend([0] * padding_to_add)

    return (
        dict(
            obss=np.array(obss),
            acts=np.array(acts),
            rews=np.array(rews),
        ),
        horizon,
    )


def observe_discrete_traj(
    cont_traj: Dict[str, Any],
    observe_times: List[int],
) -> Dict[str, Any]:
    """
    Observes the discretized trajectory based on provided sampling time indices
    - cont_traj (Dict[str, Any]): The "continuous-time" trajectory
    - horizon (int): The length of the trajectory
    - observe_times (List[int]): The sampling time indices

    """
    return {key: value[observe_times] for key, value in cont_traj.items()}


def mc_policy_iteration(
    env: Any, q_function: QFunction, observation_sampler: Any, config: SimpleNamespace
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
    iter_i = 0
    while sample_i < budget:
        traj_i = 0
        observe_times = observation_sampler.sample_time()

        cont_trajs = []
        disc_trajs = []
        ep_horizons = []
        while traj_i < num_trajs_per_update:
            # Observe "continuous-time" trajectory
            cont_traj, horizon = generate_traj(
                env,
                q_function,
                env.horizon,
                episode_i=iter_i * num_trajs_per_update + traj_i,
            )
            cont_trajs.append(cont_traj)

            # Discretize trajectory
            disc_traj = observe_discrete_traj(
                cont_traj,
                observe_times,
            )
            disc_trajs.append(disc_traj)
            ep_horizons.append(horizon)
            sample_i += len(observe_times)
            traj_i += 1

            # Check if we have broken the budget
            if sample_i >= budget:
                break

        aux = q_function.update(
            disc_trajs=disc_trajs,
            ep_horizons=ep_horizons,
            observe_times=observe_times,
            max_time=env.horizon,
            dt_sec=env.dt_sec,
        )

        if iter_i % config.log_frequency == 0:
            print(
                "Iteration {} ========================================".format(iter_i)
            )
            pprint(aux)

        iter_i += 1
