import unittest
import numpy as np

from adaptive_time.value_est import approx_integrators


class TestIntegrators(unittest.TestCase):

    def test_approx_integral(self):
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        idx = list(range(len(rewards)))
        xs = np.array([rewards, idx])
        tol = 0.001
        idxes = {}
        expected_integral = sum(rewards)
        # Do the integration.
        approx_integral = approx_integrators.approx_integrate(xs, tol, idxes)
        num_pivots = len(idxes.keys())
        print(f"approx_integral: {approx_integral} (true={expected_integral});"
              f" num_pivots: {num_pivots}/{len(rewards)}")
        self.assertAlmostEqual(approx_integral, expected_integral)
        self.assertLess(num_pivots, len(rewards))

    def test_quad_integrator_precise(self):
        """Ensure we can run them, finding the true integral."""
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        true_integral = sum(rewards)

        quad = approx_integrators.AdaptiveQuadratureIntegrator(0.0)
        quad_integral, quad_pivots = quad.integrate(rewards)
        self.assertAlmostEqual(quad_integral, true_integral)

    def test_unif_integrators_precise(self):
        """Ensure we can run them, finding the true integral."""
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        true_integral = sum(rewards)

        uniform = approx_integrators.UniformIntegrator(1)
        uniform_integral, uniform_pivots = uniform.integrate(rewards)
        self.assertAlmostEqual(uniform_integral, true_integral)

    def test_quadrature_tol(self):
        """Ensure we meet the tolerance requirement."""
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        tol = 3.0
        true_integral = sum(rewards)

        quad = approx_integrators.AdaptiveQuadratureIntegrator(tol)
        quad_integral, quad_pivots = quad.integrate(rewards)
        self.assertLessEqual(abs(quad_integral - true_integral), tol)
        print(f"quad_pivots (num={len(quad_pivots)}/{len(rewards)}): {quad_pivots}")

    def test_uniform2(self):
        """Make sure sampling every "second" point works decently.
        
        We actually see this is a little broken, we end up looking at each
        data point.
        """
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        print(rewards)
        true_integral = sum(rewards)

        unif = approx_integrators.UniformIntegrator(2)
        unif_integral, unif_pivots = unif.integrate(rewards)
        print("approx, true integral: ", unif_integral, true_integral)
        print(f"unif_pivots (num={len(unif_pivots)}/{len(rewards)}): {unif_pivots}")

    def test_uniform5(self):
        """Make sure sampling every "fifth" point works decently."""
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        print(rewards)
        true_integral = sum(rewards)

        unif = approx_integrators.UniformIntegrator(5)
        unif_integral, unif_pivots = unif.integrate(rewards)
        print("approx, true integral: ", unif_integral, true_integral)
        print(f"unif_pivots (num={len(unif_pivots)}/{len(rewards)}): {unif_pivots}")
