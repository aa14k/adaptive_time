from pprint import pprint
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from adaptive_time.q_functions import QFunction


def generate_transition(
    env: Any,
    curr_obs: Any,
    q_function: Any,
    last_observe_time: int,
    observe_time: int,
    use_action_repeat: bool,
) -> Tuple[Dict[str, Any], int]:
    """
    Generate a trajectory from the environment using the most fine-grain timescale
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from
    - last_observe_time (int): The last sample observation time
    - observe_time (int): The sample observation time
    - use_action_repeat (bool): Whether to use action repeat, or get a chance to
        re-sample the action at each time step.

    """
    done = False
    init_obs = curr_obs
    curr_act = q_function.greedy_action(curr_obs)
    while not done and last_observe_time < observe_time:
        if not use_action_repeat:
            curr_act = q_function.greedy_action(curr_obs)
        # TODO: in mountain car we do not need to sum and normalize the rewards,
        # but in general we will need to do that.
        rew, curr_obs, _, done = env.step(curr_act)
        last_observe_time += 1

    return (
        dict(
            curr_obs=init_obs,
            next_obs=curr_obs,
            act=curr_act,
            rew=rew,
            done=done,
        ),
        last_observe_time,
    )


def sarsa(
    env: Any, q_function: QFunction, observation_sampler: Any, config: SimpleNamespace
) -> Tuple[List[int], List[float]]:
    """
    Runs SARSA
    - env (Any): An environment that somewhat follows Gym API
    - q_function (Any): The Q-function to learn from
    - observation_sampler (Any): The sampler that indicates which time to observe the state
    - config (SimpleNamespace): The configuration of the learning algorithm

    config.budget: total number of samples we can observe
    config.use_action_repeat: whether to use action repeat, or get a chance to re-sample
        the action at each time step

    Returns:
    - cum_samples (List[int]): The number of total samples seen by the end of each episode.
    - ep_returns (List[float]): The returns of each (complete) episode.
    """

    budget = config.budget
    use_action_repeat = config.use_action_repeat

    sample_i = 0

    curr_obs = env.reset()
    observe_times = observation_sampler.sample_time()
    cur_ep_return = 0

    ep_returns = []
    cum_samples = []

    curr_tx, observed_time = generate_transition(
        env,
        curr_obs,
        q_function,
        -1,
        observe_times[0],
        use_action_repeat,
    )
    step_i = 1
    curr_observe_sample = observed_time

    assert not curr_tx["done"], "No samples because the observe sample exceeded horizon"

    curr_obs = curr_tx["next_obs"]
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
            use_action_repeat,
        )
        aux = q_function.update(
            curr_tx,
            next_tx,
            curr_observe_sample=curr_observe_sample,
            next_observe_sample=observed_time,
            max_time=env.horizon,
            dt_sec=env.dt_sec,
        )
        cur_ep_return += (
            curr_tx["rew"] * (observed_time - curr_observe_sample) * env.dt_sec
        )

        curr_obs = next_tx["next_obs"]
        curr_observe_sample = observed_time
        curr_tx = next_tx

        step_i += 1
        if curr_tx["done"]:
            # The episode ended.
            traj_i += 1

            curr_obs = env.reset()
            observe_times = observation_sampler.sample_time()

            cum_samples.append(sample_i + 1)
            ep_returns.append(cur_ep_return)
            cur_ep_return = 0

            curr_tx, observed_time = generate_transition(
                env,
                curr_obs,
                q_function,
                -1,
                observe_times[0],
                use_action_repeat,
            )
            step_i = 1
            curr_observe_sample = observed_time

            assert not curr_tx[
                "done"
            ], "No samples because the observe sample exceeded horizon"
            curr_obs = curr_tx["next_obs"]

        if sample_i % config.log_frequency == 0:
            print("Sample {} ====================================".format(sample_i))
            pprint(aux)
            print("Most recent 5 returns: {}".format(ep_returns[-6:-1]))

        sample_i += 1

    return cum_samples, ep_returns
