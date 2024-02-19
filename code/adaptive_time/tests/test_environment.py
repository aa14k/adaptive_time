import unittest
import logging
import random
import numpy as np

from adaptive_time import environment


# Get logger
logger = logging.getLogger("__test_environment__")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)


class MountainCarTest(unittest.TestCase):

    def test_run_env(self):
        rng = random.Random(13)
        env = environment.MountainCar(horizon_sec=10)
        rewards = []
        states = []
        hs = []
        dones = []
        # Collect actual data.
        logger.info("Running environment.")
        for s in range(11):
            ret_vals = env.step(rng.choice([0, 2]))
            logger.info("returned values: %r", ret_vals)
            reward, state, h_and_h_disc, done = ret_vals
            rewards.append(reward)
            states.append(state)
            hs.append(h_and_h_disc)
            dones.append(done)

        # The expected data.
        expected_rewards = [-1.0] * 11
        expected_states = np.array(
            [
                [-0.49917684, 0.00082316],
                [-0.49753669, 0.00164016],
                [-0.497091797, 0.000444889747],
                [-0.4978455, -0.0007537],
                [-0.49979216, -0.00194666],
                [-0.50291722, -0.00312506],
                [-0.50719729, -0.00428007],
                [-0.51260032, -0.00540303],
                [-0.51908583, -0.00648551],
                [-0.52660518, -0.00751935],
                [-0.53310198, -0.0064968],
            ]
        )
        expected_hs = [(float(i + 1), i) for i in range(11)]
        expected_dones = [False] * 9 + [True] * 2

        # Check correctness.
        self.assertEqual(rewards, expected_rewards)
        np.testing.assert_allclose(states, expected_states)
        self.assertEqual(hs, expected_hs)
        self.assertEqual(dones, expected_dones)

    def test_dt_equivalence(self):
        """Test that the environment behaves the same with different dt."""
        rng = random.Random(13)
        env_coarse = environment.MountainCar(horizon_sec=10, dt_sec=1.0)
        env_fine = environment.MountainCar(horizon_sec=10, dt_sec=0.2)
        logger.info("Running environment.")

        for second in range(12):
            action = rng.choice([-1, 1])

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

    def test_create_trajectories(self):
        """Test that we can create trajectories."""
        rng = random.Random(13)
        policy = lambda state: rng.randint(0, 2)
        num_trajectories = 3
        horizon_sec = 10
        dt_sec = 0.001
        num_data_points_per_traj = int(horizon_sec / dt_sec)
        trajectories = environment.create_trajectories(
            num_trajectories, policy, horizon_sec, dt_sec
        )
        self.assertEqual(len(trajectories), num_trajectories)
        for states, rewards in trajectories:
            self.assertEqual(len(states), num_data_points_per_traj + 1)
            self.assertEqual(len(rewards), num_data_points_per_traj)


class CorridorTest(unittest.TestCase):

    def test_run_env_dt(self):
        env = environment.Corridor(length=5, horizon_sec=10., dt_sec=1.)
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
        env_coarse = environment.Corridor(horizon_sec=10, dt_sec=1.0)
        env_fine = environment.Corridor(horizon_sec=10, dt_sec=0.2)
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
