from pprint import pprint

import argparse
import json
import numpy as np

from code.environment import MountainCar
from code.monte_carlo import mc_policy_iteration
from code.sarsa import sarsa
from code.q_functions import MountainCarTileCodingQ
from code.utils import parse_dict


def main(args):
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    config = parse_dict(config_dict)
    pprint(config)

    np.random.seed(config.seed)

    # TODO: Replace this with regulator
    class MockSampler:
        def __init__(self, num_steps: int):
            self.num_steps = num_steps

        def sample_time(self):
            return np.arange(0, self.num_steps, 2)

    # TODO: Better way to construct these objects
    observation_sampler = MockSampler(config.env_kwargs.horizon_sec)
    q_function = None
    env = None
    if config.env == "mountain_car":
        q_function = MountainCarTileCodingQ(config.agent_config)
        env = MountainCar(**vars(config.env_kwargs))

    if config.agent_config.update_rule == "monte_carlo":
        mc_policy_iteration(
            env=env,
            q_function=q_function,
            observation_sampler=observation_sampler,
            config=config,
        )
    elif config.agent_config.update_rule == "sarsa":
        sarsa(
            env=env,
            q_function=q_function,
            observation_sampler=observation_sampler,
            config=config,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The experiment configuration path",
    )
    args = parser.parse_args()

    main(args)
