from pprint import pprint
from types import SimpleNamespace
from typing import Dict

import argparse
import json
import numpy as np

from environment import MountainCar
from monte_carlo import mc_policy_iteration
from sarsa import sarsa
from q_functions import MountainCarTileCodingQ


def parse_dict(d: Dict) -> SimpleNamespace:
    """
    Parse dictionary into a namespace.
    Reference: https://stackoverflow.com/questions/66208077/how-to-convert-a-nested-python-dictionary-into-a-simple-namespace

    :param d: the dictionary
    :type d: Dict
    :return: the namespace version of the dictionary's content
    :rtype: SimpleNamespace

    """
    x = SimpleNamespace()
    _ = [
        setattr(x, k, parse_dict(v)) if isinstance(v, dict) else setattr(x, k, v)
        for k, v in d.items()
    ]
    return x


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
            return np.arange(self.num_steps)

    # TODO: Better way to construct these objects
    observation_sampler = MockSampler(config.env_kwargs.horizon)
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
