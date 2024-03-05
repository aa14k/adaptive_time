import unittest
import numpy as np

from adaptive_time.environments import cartpole2


class Cartpole2Test(unittest.TestCase):

    def test_discrete_env(self):
        env = cartpole2.CartPoleEnv(discrete_reward=True)
        state, _ = env.reset()
        env.action_space.seed(0)
        while True:
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            self.assertEqual(reward, 1.0)
            if done:
                break

    def test_continuous_env(self):
        env = cartpole2.CartPoleEnv(discrete_reward=False)
        state, _ = env.reset()
        env.action_space.seed(0)
        rewards = []
        while True:
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        self.assertTrue(np.mean(rewards) < 0.99)



