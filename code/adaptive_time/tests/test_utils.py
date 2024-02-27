import unittest
import logging
import random
import numpy as np

from adaptive_time import utils


# Get logger
logger = logging.getLogger("__test_mountaincar_tilecoding__")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)


class Test(unittest.TestCase):
    def test_discounted_returns(self):
        traj = [(0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)]
        gamma = 0.9
        expected = [2.71, 1.9, 1.0]
        self.assertEqual(utils.discounted_returns(traj, gamma), expected)

    def test_total_returns(self):
        traj = [(0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 2, 0)]
        expected = [4.0, 3.0, 2.0]
        np.testing.assert_allclose(utils.total_returns(traj), expected)
