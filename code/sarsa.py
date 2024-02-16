from pprint import pprint
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from q_functions import QFunction


# TODO: Modify generation
def generate_transition(
    env: Any,
    curr_obs: Any,
    q_function: Any,
    observe_time: int,
) -> Tuple[Dict[str, Any], int]:
    """
    Generate a trajectory from the environment using the most fine-grain timescale
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from
    - observe_time (int): The time to sample, relative to current time

    """
    done = False
    curr_time = 0
    while not done and curr_time < observe_time:
        curr_act = q_function.greedy_action(curr_obs)
        rew, curr_obs, _, done = env.step(curr_act)
        curr_time += 1

    return dict(
        next_obs=curr_obs,
        act=curr_act,
        rew=rew,
        done=done,
    ), curr_time


def sarsa(
    env: Any, q_function: QFunction, observation_sampler: Any, config: SimpleNamespace
):
    """
    Runs SARSA
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from
    - observation_sampler (Any): The sampler that indicates which time to observe the state
    - config (SimpleNamespace): The configuration of the learning algorithm

    config.budget: total number of samples we can observe

    """

    budget = config.budget

    sample_i = 0
    step_i = 0

    curr_obs = env.reset()
    observe_times = observation_sampler.sample_time()
    curr_observe_sample = 0
    ep_returns = [0]

    while sample_i < budget:
        traj_i = 0
        next_observe_sample = observe_times[step_i]

        # TODO: Modify to account edge case (i.e. sample t=0)
        # Observe next observation based on suggested observation time
        disc_tx, observed_time = generate_transition(
            env,
            curr_obs,
            q_function,
            next_observe_sample
        )
        aux = q_function.update(
            curr_obs=curr_obs,
            **disc_tx,
            curr_observe_sample=curr_observe_sample,
            next_observe_sample=observed_time,
            max_time=env.horizon,
        )

        curr_obs = disc_tx["next_obs"]

        ep_returns[-1] += disc_tx["rew"] * (observed_time - curr_observe_sample)
        curr_observe_sample = observed_time

        step_i += 1
        if disc_tx["done"]:
            step_i = 0
            traj_i += 1
            ep_returns.append(0)
            curr_observe_sample = 0
            curr_obs = env.reset()
            print("RESET")
            observe_times = observation_sampler.sample_time()

        if sample_i % config.log_frequency == 0:
            print("Sample {} ====================================".format(sample_i))
            pprint(aux)
            print("Most recent 5 returns: {}".format(ep_returns[-6:-1]))

        sample_i += 1
