import unittest
import logging
import random
import numpy as np

from adaptive_time.environments import gym_mountain_car

# Get logger
logger = logging.getLogger("__test_environment__")
logger.setLevel(logging.INFO)

if (logger.hasHandlers()):
    logger.handlers.clear()

c_handler = logging.StreamHandler()
logger.addHandler(c_handler)


class MountainCarTest(unittest.TestCase):

    def test_run_env(self):
        rng = random.Random(13)
        env = gym_mountain_car.MountainCarEnv(dt=0.1)
        env.reset(seed=13)
        rewards = []
        states = []
        hs = []
        dones = []
        # Collect actual data.
        logger.info("Running environment.")
        for s in range(11):
            ret_vals = env.step(rng.choice([0, 2]))
            logger.info("returned values: %r", ret_vals)
            state, reward, done, truncated, info = ret_vals

    def test_dt_equivalence(self):
        """Test that the environment behaves the same with different dt."""
        rng = random.Random(13)
        env_coarse = gym_mountain_car.MountainCarEnv(dt=1.0)
        env_fine = gym_mountain_car.MountainCarEnv(dt=0.2)
        logger.info("Running environments.")

        env_coarse.reset(seed=13)
        env_fine.reset(seed=13)

        for second in range(12):
            action = rng.choice([0, 1])  # not 2, so we don't turn around

            total_fine_reward = 0
            for sub_second in range(5):
                # Fine environment.
                (state_fine, reward_fine, done_fine, truncated,
                 info) = env_fine.step(action)
                total_fine_reward += reward_fine
                logger.info(
                    "s=%r/%r;  FINE: %r, %r, %r",
                    second,
                    sub_second,
                    reward_fine,
                    state_fine,
                    done_fine,
                )

            # Coarse environment.
            (state_coarse, reward_coarse, done_coarse, truncated,
                 info) = env_coarse.step(action)
            logger.info(
                "s=%r;  COARSE: %r, %r, %r",
                second,
                reward_coarse,
                state_coarse,
                done_coarse,
            )

            self.assertEqual(reward_coarse, total_fine_reward)
            np.testing.assert_allclose(
                state_coarse, state_fine, atol=0.000005, rtol=0.015)
            self.assertEqual(done_coarse, done_fine)

