import unittest
import random

from adaptive_time import environments


class UtilitiesTest(unittest.TestCase):

    def test_create_trajectories(self):
        """Test that we can create trajectories."""
        rng = random.Random(13)
        policy = lambda state: rng.randint(0, 2)
        num_trajectories = 3
        horizon_sec = 10
        dt_sec = 0.001
        num_data_points_per_traj = int(horizon_sec / dt_sec)
        env = lambda: environments.MountainCar(
            horizon_sec=horizon_sec, dt_sec=dt_sec)
        trajectories = environments.create_trajectories(
            env, num_trajectories, policy
        )
        self.assertEqual(len(trajectories), num_trajectories)
        for states, rewards in trajectories:
            self.assertEqual(len(states), num_data_points_per_traj + 1)
            self.assertEqual(len(rewards), num_data_points_per_traj)

