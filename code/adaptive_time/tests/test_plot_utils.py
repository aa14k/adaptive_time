import unittest
import logging
import random
import numpy as np

from adaptive_time import plot_utils
from parameterized import parameterized


# Get logger
logger = logging.getLogger("__test_mountaincar_tilecoding__")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)


class Test(unittest.TestCase):

    def test_pad_combine(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        c = np.array([6, 7, 8, 9])
        res = plot_utils.pad_combine([a, b, c], pad_value=0)
        expected = np.array([
            [1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 7, 8, 9]
        ]).T
        np.testing.assert_array_equal(res, expected)


    def test_interpolate_and_stack(self):
        dict_of_multiple_runs = {
            "m1": [
                {"x": [0, 4], "y": [0, 8]},
                {"x": [0, 2, 4], "y": [0, 4, 8]},
            ],
            "m2": [
                {"x": [0, 6], "y": [0, 6]},
                {"x": [0, 3, 6], "y": [1, 4, 7]},
            ]
        }
        all_x_values, interpolated = plot_utils.interpolate_and_stack(
                dict_of_multiple_runs, "x", "y", right=0)
        expected_all_x_values = np.array([0, 1, 2, 3, 4, 5, 6])
        expected_interpolated = {
            "m1": np.array([
                    [0, 2, 4, 6, 8, 0, 0],
                    [0, 2, 4, 6, 8, 0, 0],
                ]),
            "m2": np.array([
                    [0, 1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6, 7],
                ])
        }
        np.testing.assert_array_equal(
            all_x_values, expected_all_x_values)
        self.assertEqual(
            interpolated.keys(), expected_interpolated.keys())

        for k in interpolated.keys():
            np.testing.assert_array_equal(
                interpolated[k], expected_interpolated[k])
