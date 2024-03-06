import unittest
import numpy as np

from parameterized import parameterized

from adaptive_time.value_est import approx_integrators


class TestIntegrators(unittest.TestCase):
    """Test the approximators for the integral of a reward sequence.

    Most of these tests will use the same example data, just so it's easier to
    think about. The reward sequence is:

        dat: [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    The true integral is 26.
    """

    def test_quad_integrator_precise(self):
        """Ensure we can run them, finding the true integral."""
        rewards = [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        true_integral = sum(rewards)

        quad = approx_integrators.AdaptiveQuadratureIntegrator(0.0)
        quad_integral, quad_pivots = quad.integrate(rewards)
        self.assertAlmostEqual(quad_integral, true_integral)


    def test_quadrature_tol(self):
        """Ensure we meet the tolerance requirement."""
        rewards = [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        tol = 3.0   # A high tolerance so we don't have to use all data points.
        true_integral = sum(rewards)

        quad = approx_integrators.AdaptiveQuadratureIntegrator(tol)
        quad_integral, quad_pivots = quad.integrate(rewards)
        self.assertLessEqual(abs(quad_integral - true_integral), tol)
        print(f"quad_pivots (num={len(quad_pivots)}/{len(rewards)}): {quad_pivots}")

    def test_unif_integrators_precise(self):
        """Ensure u1 finds the true integral, using all points."""
        rewards = [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        print(np.array([rewards, np.arange(len(rewards))]))
        true_integral = sum(rewards)

        uniform = approx_integrators.UniformlySpacedIntegrator(1)
        uniform_integral, uniform_pivots = uniform.integrate(rewards)
        self.assertAlmostEqual(uniform_integral, true_integral)
        self.assertEqual(len(uniform_pivots), len(rewards))

    def test_uniform2(self):
        """Make sure sampling every second point towards a trapezoid approx works.
        
        Recall, the data is

        ```
            dat: [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
            idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ```

        This corresponds to doing estimation by forming trapezoids between
        idxes 0 and 2, 2 and 4, 4 and 6, 6 and 8, 8 and 9 (since there is no 10).
        This leads to trapezoid approximations of the partial sums of:
        
        ```
            approx:    0+0+1=1, 1+2+3=6, 3+4+4=11, 4+4+4=12, 4+4=8,
            through:     1.5       6       10.5      12       8         
        ```
        
        But note that this double counts each intermediate point, so we just
        subtract these out. Therefore we except the approximation to be
        `1.5 + 6 + 10.5 + 12 + 8 - (1+3+4+4) = 38-12 = 26`.
        """
        rewards = [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        print(np.array([rewards, np.arange(len(rewards))]))
        expected_approx_integral = 26
        expected_num_pivots = 6

        unif = approx_integrators.UniformlySpacedIntegrator(2)
        unif_integral, unif_pivots = unif.integrate(rewards)
        print("approx vs expected integral: ", unif_integral, expected_approx_integral)
        print(f"unif_pivots (num={len(unif_pivots)}/{len(rewards)}): {unif_pivots}")

        self.assertEqual(unif_integral, expected_approx_integral)
        self.assertEqual(len(unif_pivots), expected_num_pivots)

    def test_uniform5(self):
        """Make sure sampling every fifth point works.
        
        See `test_uniform2` for a detailed explanation of the expected result.
        Here I do a quick calculation to get the expected result.

        ```
        Orig data:
            dat: [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
            idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        Workings:
            1. 0->5: (0+4)/2 * 6 = 12
            2. 5->9: (4+4)/2 * 5 = 20
            total: 12+20 - 4 = 28
        ```
        """
        rewards = [0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        print(np.array([rewards, np.arange(len(rewards))]))
        expected_approx = 28
        expected_pivots = 3

        unif = approx_integrators.UniformlySpacedIntegrator(5)
        unif_integral, unif_pivots = unif.integrate(rewards)
        print("approx vs expected integral: ", unif_integral, expected_approx)
        print(f"unif_pivots (num={len(unif_pivots)}/{len(rewards)}): {unif_pivots}")

        self.assertEqual(unif_integral, expected_approx)
        self.assertEqual(len(unif_pivots), expected_pivots)

    # def test_on_real_data(self):
    #     """Use our own data."""
    #     import pickle
    #     load_data_from = "many_good_trajs.pkl"

    #     with open(load_data_from, "rb") as f:
    #         data = pickle.load(f)
    #     total_rewards, reward_sequences, traj_lengths = data

    #     samplers_tried = dict(
    #         q100=approx_integrators.AdaptiveQuadratureIntegrator(tolerance=100),
    #         q10=approx_integrators.AdaptiveQuadratureIntegrator(tolerance=10),
    #         q1=approx_integrators.AdaptiveQuadratureIntegrator(tolerance=1),
    #         u1=approx_integrators.UniformlySpacedIntegrator(1),
    #         u10=approx_integrators.UniformlySpacedIntegrator(10),
    #         u100=approx_integrators.UniformlySpacedIntegrator(100),
    #         u1000=approx_integrators.UniformlySpacedIntegrator(1000),
    #         u10000=approx_integrators.UniformlySpacedIntegrator(10000),
    #     )

    #     approx_integrals = {}
    #     num_pivots = {}

    #     for sampler_name, sampler in samplers_tried.items():
    #         print("sampler_name:", sampler_name)
    #         approx_integrals[sampler_name] = []
    #         num_pivots[sampler_name] = []
    #         for idx, reward_seq in enumerate(reward_sequences[:1]):
    #             integral, all_pivots = sampler.integrate(reward_seq)
    #             approx_integrals[sampler_name].append(integral)
    #             num_pivots[sampler_name].append(len(all_pivots))
    #     print(approx_integrals)

    def test_approx_integral(self):
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        idx = list(range(len(rewards)))
        xs = np.array([rewards, idx])
        tol = 0.001
        idxes = {}
        expected_integral = sum(rewards)
        # Do the integration.
        approx_integral = approx_integrators.adaptive_approx_integrate(xs, tol, idxes)
        num_pivots = len(idxes.keys())
        print(f"approx_integral: {approx_integral} (true={expected_integral});"
              f" num_pivots: {num_pivots}/{len(rewards)}")
        self.assertAlmostEqual(approx_integral, expected_integral)
        self.assertLess(num_pivots, len(rewards))

    @parameterized.expand([
        (1,), (2,), (3,), (4,), (5,)
    ])
    def test_quadrature_base_cases(self, length):
        rewards = [0, 1, 2, 4, 8, 16]
        rewards = rewards[:length]
        tol = 0.0
        true_integral = sum(rewards)

        quad = approx_integrators.AdaptiveQuadratureIntegrator(tol)
        quad_integral, quad_pivots = quad.integrate(rewards)
        self.assertLessEqual(abs(quad_integral - true_integral), tol)
