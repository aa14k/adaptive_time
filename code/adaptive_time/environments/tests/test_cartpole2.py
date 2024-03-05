import unittest
import numpy as np

from adaptive_time.environments import cartpole2


class Cartpole2Test(unittest.TestCase):

    def test_discrete_env(self):
        env = cartpole2.CartPoleEnv(discrete_reward=True)
        state, _ = env.reset()
        env.action_space.seed(0)
        num_steps = 0
        max_steps = 10   # To prevent infinite loops.
        while num_steps < max_steps:
            num_steps += 1
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
        num_steps = 0
        max_steps = 10   # To prevent infinite loops.
        while num_steps < max_steps:
            num_steps += 1
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        self.assertTrue(np.mean(rewards) < 0.999)



