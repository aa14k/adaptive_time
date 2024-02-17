import unittest
import logging
import random
import numpy as np

from code.q_functions import MountainCarTileCodingQ
from code.utils import parse_dict


# Get logger
logger = logging.getLogger("__test_mountaincar_tilecoding__")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)


class Test(unittest.TestCase):
    def test_mc_update(self):
        q_config = {
            "q_function": "tile_coding",
            "update_rule": "monte_carlo",
            "param_init_mean": 1,
            "param_init_std": 0.1,
            "iht_size": 4096,
            "num_tiles": 2,
            "num_tilings": 2,
            "learning_rate": 3e-4,
            "action_space": [-1, 0, 1],
            "seed": 43,
        }
        q_function = MountainCarTileCodingQ(parse_dict(q_config))
        q_function.parameters = np.ones(q_function.parameters.shape)

        obss = np.arange(10).reshape((5, 2)) * 0.005
        features = [q_function.get_feature(obs) for obs in obss]
        acts = [0, 1, 2, 0, 1]
        rews = [0, 0, 0, 0, 1]
        max_time = 5
        observe_times = [0, 1, 2]

        from pprint import pprint
        pprint(features)
        self.fail()

        # self.assertEqual(rewards, expected_rewards)
        # self.assertTrue(np.allclose(states, expected_states))
