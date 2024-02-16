import unittest
import logging
import random

from code import environment

# Get logger
logger = logging.getLogger("__environment__")
logger.setLevel(logging.INFO)


class Test(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(4, 4)

    def test_create_env(self):
        rng = random.Random(13)
        env = environment.MountainCar(horizon=10)
        # for s in range(11):
        #     ret = env.step(rng.choice([-1, 1]))
        #     logging.info("returned: %r", ret)
        # reward, state, h, done = env.step(rng.choice([-1, 1]))
        # logger.info(
        #     "r: %.1f;  pos: %.2f;  h: %d;  done: %r.",
        #     reward, state, h, done)
