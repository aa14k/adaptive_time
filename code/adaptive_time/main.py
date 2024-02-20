from pprint import pprint

import argparse
import json
import numpy as np

from adaptive_time.environments import MountainCar, Corridor
from adaptive_time.monte_carlo import mc_policy_iteration
from adaptive_time.samplers import UniformSampler, AdaptiveQuadratureSampler
from adaptive_time.sarsa import sarsa
from adaptive_time.q_functions import Q
from adaptive_time.utils import parse_dict
from adaptive_time import features


def setup(config):
    # TODO: Better way to construct these objects

    # Construct Q-function and environment
    q_function = None
    env = None

    if config.env == "mountain_car":
        feature_extractor = features.MountainCarTileCoder(
            iht_size=getattr(config, "iht_size", 4096),
            num_tilings=getattr(config, "num_tilings", 8),
            num_tiles=getattr(config, "num_tiles", 8),
        )
        env = MountainCar(**vars(config.env_kwargs))
    elif config.env == "corridor":
        if config.env_kwargs["dt_sec"] != 1.0:
            raise ValueError(
                "The Tabular feature extractor used below only supports "
                f"dt_sec=1.0, but dt_sec is {config.env_kwargs['dt_sec']}.")
        # TODO: use a tilecoder or something instead to deal with different dt_sec.
        feature_extractor = features.Tabular(config.env_kwargs["length"])
        env = Corridor(**vars(config.env_kwargs))
    else:
        raise NotImplementedError

    q_function = Q(feature_extractor, config.agent_config)

    # Construct observation sampler
    if config.sampler_config.sampler == "uniform":
        observation_sampler = UniformSampler(
            env.horizon - 1,
            config.sampler_config.sampler_kwargs.spacing,
        )
    elif config.sampler_config.sampler == "adaptive_quadrature":
        observation_sampler = AdaptiveQuadratureSampler(
            dt=env.dt_sec,
            num_steps=env.horizon - 1,
            tolerance_init=config.sampler_config.sampler_kwargs.tolerance_init,
            integral_rule=config.sampler_config.sampler_kwargs.integral_rule,
            update_when_best=config.sampler_config.sampler_kwargs.update_when_best,
        )
    else:
        raise NotImplementedError

    return q_function, env, observation_sampler


def run(config_dict):

    config = parse_dict(config_dict)
    pprint(config)

    np.random.seed(config.seed)

    q_function, env, observation_sampler = setup(config)

    if config.agent_config.update_rule.endswith("monte_carlo"):
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

    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    run(config_dict)
