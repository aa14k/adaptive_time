import unittest
import numpy as np

from parameterized import parameterized

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
        print(np.array([rewards, np.arange(len(rewards))]))
        true_integral = sum(rewards)

        uniform = approx_integrators.UniformlySpacedIntegrator(1)
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

    def test_uniform2_odd(self):
        """Make sure sampling every "second" point works decently.
        
        We actually see this is a little broken, we end up looking at each
        data point.
        """
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        print(np.array([rewards, np.arange(len(rewards))]))
        true_integral = sum(rewards)

        unif = approx_integrators.UniformlySpacedIntegrator(2)
        unif_integral, unif_pivots = unif.integrate(rewards)
        print("approx, true integral: ", unif_integral, true_integral)
        print(f"unif_pivots (num={len(unif_pivots)}/{len(rewards)}): {unif_pivots}")

    def test_uniform2_even(self):
        """Make sure sampling every "second" point works decently.
        
        We actually see this is a little broken, we end up looking at each
        data point.
        """
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4]
        print(np.array([rewards, np.arange(len(rewards))]))
        true_integral = sum(rewards)

        unif = approx_integrators.UniformlySpacedIntegrator(2)
        unif_integral, unif_pivots = unif.integrate(rewards)
        print("approx, true integral: ", unif_integral, true_integral)
        print(f"unif_pivots (num={len(unif_pivots)}/{len(rewards)}): {unif_pivots}")

    def test_uniform5(self):
        """Make sure sampling every "fifth" point works decently."""
        rewards = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]
        print(np.array([rewards, np.arange(len(rewards))]))
        true_integral = sum(rewards)

        unif = approx_integrators.UniformlySpacedIntegrator(5)
        unif_integral, unif_pivots = unif.integrate(rewards)
        print("approx, true integral: ", unif_integral, true_integral)
        print(f"unif_pivots (num={len(unif_pivots)}/{len(rewards)}): {unif_pivots}")

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
