import unittest
import numpy as np

from adaptive_time import value_est


class Test(unittest.TestCase):

    def test_approx_integral(self):
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        idx = list(range(len(rewards)))
        xs = np.array([rewards, idx])
        tol = 0.001
        idxes = {}
        expected_integral = sum(rewards)
        # Do the integration.
        approx_integral = value_est.approx_integrate(xs, tol, idxes)
        num_pivots = len(idxes.keys())
        print(f"approx_integral: {approx_integral} (true={expected_integral});"
              f" num_pivots: {num_pivots}/{len(rewards)}")
        self.assertAlmostEqual(approx_integral, expected_integral)
        self.assertLess(num_pivots, len(rewards))
