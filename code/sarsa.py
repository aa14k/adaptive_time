from pprint import pprint
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from code.q_functions import QFunction


def generate_transition(
    env: Any,
    curr_obs: Any,
    q_function: Any,
    last_observe_time: int,
    observe_time: int,
) -> Tuple[Dict[str, Any], int]:
    """
    Generate a trajectory from the environment using the most fine-grain timescale
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from
    - last_observe_time (int): The last sample observation time
    - observe_time (int): The sample observation time

    """
    done = False
    while not done and last_observe_time < observe_time:
        curr_act = q_function.sample_action(curr_obs)
        rew, curr_obs, _, done = env.step(curr_act)
        last_observe_time += 1

    return (
        dict(
            obs=curr_obs,
            act=curr_act,
            rew=rew,
            done=done,
        ),
        last_observe_time,
    )


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

    curr_obs = env.reset()
    observe_times = observation_sampler.sample_time()
    ep_returns = [0]

    curr_tx, observed_time = generate_transition(
        env,
        curr_obs,
        q_function,
        -1,
        observe_times[0],
    )
    step_i = 1
    curr_observe_sample = observed_time

    assert not curr_tx["done"], "No samples because the observe sample exceeded horizon"

    curr_obs = curr_tx["obs"]
    while sample_i < budget:
        traj_i = 0

        if step_i == len(observe_times):
            next_observe_sample = env.horizon
        else:
            next_observe_sample = observe_times[step_i]

        # Observe next observation based on suggested observation time
        next_tx, observed_time = generate_transition(
            env,
            curr_obs,
            q_function,
            curr_observe_sample,
            next_observe_sample,
        )
        aux = q_function.update(
            curr_tx,
            next_tx,
            curr_observe_sample=curr_observe_sample,
            next_observe_sample=observed_time,
            max_time=env.horizon,
        )
        ep_returns[-1] += curr_tx["rew"] * (observed_time - curr_observe_sample)

        curr_obs = next_tx["obs"]
        curr_observe_sample = observed_time
        curr_tx = next_tx

        step_i += 1
        if curr_tx["done"]:
            traj_i += 1

            curr_obs = env.reset()
            observe_times = observation_sampler.sample_time()
            ep_returns = [0]

            curr_tx, observed_time = generate_transition(
                env,
                curr_obs,
                q_function,
                -1,
                observe_times[0],
            )
            step_i = 1
            curr_observe_sample = observed_time

            assert not curr_tx[
                "done"
            ], "No samples because the observe sample exceeded horizon"

        if sample_i % config.log_frequency == 0:
            print("Sample {} ====================================".format(sample_i))
            pprint(aux)
            print("Most recent 5 returns: {}".format(ep_returns[-6:-1]))

        sample_i += 1
