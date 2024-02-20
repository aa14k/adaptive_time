import unittest
import logging
import random
import numpy as np

from adaptive_time.environments import corridor


# Get logger
logger = logging.getLogger("__test_environment__")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)


class CorridorTest(unittest.TestCase):

    def test_run_env_dt(self):
        env = corridor.Corridor(length=5, horizon_sec=10., dt_sec=1.)
        rewards = []
        states = []
        hs = []
        dones = []
        # Collect actual data.
        logger.info("Running environment.")
        for s in range(11):
            ret_vals = env.step(2)
            logger.info("returned values: %r", ret_vals)
            reward, state, h_and_h_disc, done = ret_vals
            rewards.append(reward)
            states.append(state)
            hs.append(h_and_h_disc)
            dones.append(done)

        # The expected data; we should reach the goal at step 4.
        expected_rewards = [-1.0] * 3 + [0.0] * 8
        expected_states = list(range(1, 4)) + [4] * 8
        expected_hs = [(float(i + 1), i) for i in range(11)]
        expected_dones = [False] * 9 + [True] * 2

        # Check correctness.
        self.assertEqual(rewards, expected_rewards)
        np.testing.assert_allclose(states, expected_states)
        self.assertEqual(hs, expected_hs)
        self.assertEqual(dones, expected_dones)

    # TODO: make this test parametrized and use it for both environments.
    def test_dt_equivalence(self):
        """Test that the environment behaves the same with different dt."""
        rng = random.Random(13)
        env_coarse = corridor.Corridor(horizon_sec=10, dt_sec=1.0)
        env_fine = corridor.Corridor(horizon_sec=10, dt_sec=0.2)
        logger.info("Running environment.")

        for second in range(12):
            action = rng.choice([0, 1, 2])

            for sub_second in range(5):
                # Fine environment.
                reward_fine, state_fine, hs_fine, done_fine = env_fine.step(action)
                logger.info(
                    "s=%r/%r;  FINE: %r, %r, %r, %r",
                    second,
                    sub_second,
                    reward_fine,
                    state_fine,
                    hs_fine,
                    done_fine,
                )

            # Coarse environment.
            reward_coarse, state_coarse, hs_coarse, done_coarse = env_coarse.step(
                action
            )
            logger.info(
                "s=%r;  COARSE: %r, %r, %r, %r",
                second,
                reward_coarse,
                state_coarse,
                hs_coarse,
                done_coarse,
            )

            self.assertEqual(reward_coarse, reward_fine)
            np.testing.assert_allclose(state_coarse, state_fine, atol=0.000005, rtol=0.015)
            self.assertAlmostEqual(hs_coarse[0], hs_fine[0])
            self.assertEqual(done_coarse, done_fine)
