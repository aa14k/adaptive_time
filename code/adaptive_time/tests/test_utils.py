import unittest
import logging
import random
import numpy as np

from adaptive_time import utils
from parameterized import parameterized


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

    def test_argmax_basic(self):
        x = np.array([1, 2, 3, 4])
        max_idx, probs = utils.argmax_with_probs(
            x, calc_action=True, calc_probs=True)
        self.assertEqual(max_idx, 3)
        np.testing.assert_almost_equal(probs, np.array([0.0, 0.0, 0.0, 1.]))

    # def test_argmax_fake_2d(self):
    #     """Same as basic, but there is an ignored dimension"""
    #     x = np.array([[1], [2], [3], [4]])
    #     max_idx, probs = utils.argmax(x, return_probs=True)
    #     self.assertEqual(max_idx, 3)
    #     # np.testing.assert_almost_equal(probs, np.array([0.0, 0.0, 0.0, 1.]))

    def test_argmax_equals(self):
        x = np.array([4, 3, 3, 4])
        max_idx, probs = utils.argmax_with_probs(
            x, calc_action=True, calc_probs=True)
        self.assertIn(max_idx, [0, 3])
        np.testing.assert_almost_equal(probs, np.array([0.5, 0.0, 0.0, 0.5]))

    @parameterized.expand([
        (0.0, [0, 1], [0, 1]),
        (1.0, [0, 1], [0.5, 0.5]),
        (0.1, [1, 1, 0, 0.5], [0.475, 0.475, 0.025, 0.025]),
    ])
    def test_eps_greedy_policy_probs(self, eps, qs, expected):
        qs_ar = np.array(qs)
        probs = utils.eps_greedy_policy_probs(eps, qs_ar)
        expected_ar = np.array(expected)
        np.testing.assert_almost_equal(probs, expected_ar)

    @parameterized.expand([
        (0.0, [0, 2], 2),
        (1.0, [0, 2], 1),
        (0.1, [1, 1, 0, 0.5], 0.9625),
    ])
    def test_v_from_eps_greedy_q(self, eps, qs, expected):
        qs_ar = np.array(qs)
        probs = utils.v_from_eps_greedy_q(eps, qs_ar)
        np.testing.assert_almost_equal(probs, expected)

    @parameterized.expand([
        ("/Users/t/adaptive_time/code/adaptive_time", "/Users/t/adaptive_time"),
        ("/Users/t/adaptive_time/", "/Users/t/adaptive_time"),
    ])
    def test_find_root_directory(self, orig_path, expected):
        self.assertEqual(utils.find_root_directory(orig_path), expected)

    def test_find_root_directory_bad(self):
        path = "/Users/t/otherproject/code/bla/"
        self.assertRaises(ValueError, utils.find_root_directory, path)
