import unittest
import logging
import random
import numpy as np

from code import environment


# Get logger
logger = logging.getLogger('__test_environment__')
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)

class Test(unittest.TestCase):

    def test_run_env(self):
        rng = random.Random(13)
        env = environment.MountainCar(horizon=10)
        rewards = []
        states = []
        hs = []
        dones = []
        # Collect actual data.
        logger.info("Running environment.")
        for s in range(11):
            ret_vals = env.step(rng.choice([0, 2]))
            logger.info("returned values: %r", ret_vals)
            reward, state, h, done = ret_vals
            rewards.append(reward)
            states.append(state)
            hs.append(h)
            dones.append(done)

        # The expected data.
        expected_rewards = [-1.0] * 11
        expected_states = np.array([
            [-0.49917684,  0.00082316],
            [-0.49753669,  0.00164016],
            [-0.497091797,  0.000444889747],
            [-0.4978455, -0.0007537],
            [-0.49979216, -0.00194666],
            [-0.50291722, -0.00312506],
            [-0.50719729, -0.00428007],
            [-0.51260032, -0.00540303],
            [-0.51908583, -0.00648551],
            [-0.52660518, -0.00751935],
            [-0.53310198, -0.0064968 ],
        ])
        expected_hs = list(range(11))
        expected_dones = [False] * 9 + [True] * 2

        # Check correctness.
        self.assertEqual(rewards, expected_rewards)
        self.assertTrue(np.allclose(states, expected_states))
        self.assertEqual(hs, expected_hs)
        self.assertEqual(dones, expected_dones)
        

            