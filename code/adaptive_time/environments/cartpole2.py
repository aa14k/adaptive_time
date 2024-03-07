"""
This is a modified version of the Gymnasium CartPole environment.

Changes:
1. We don't terminate the episode if the pole angle is greater than ±12°, nor
    if the cart position is greater than ±2.4. (This is because we want to make
    sure the agent experiences 0 rewards as well, but it's hard to do that if
    the episode is terminated.)
# 1. We don't terminate the episode if the pole angle is greater than ±12°, nor
#     We also do not terminate if cart position is greater than ±2.4, though
#     we bound the state space to this range. (This is because we want to make
#     sure the agent experiences 0 rewards as well, but it's hard to do that if
#     the episode is terminated.)
2. The state can be initialized with a custom state.
3. There are two reward variants; the original, and one that slowly varies
    with the state. This is so that the reward function is not trivially
    learnable with adaptive quadrature methods.
4. We truncate the episode if the episode length is greater than 20_000.

Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


gym.register(
    id="CartPole-OURS-v2",
    entry_point="adaptive_time.environments.cartpole2:CartPoleEnv",
    vector_entry_point=None,
    max_episode_steps=20_000,
    reward_threshold=None,
)


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
    (array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})

    ```

    ## Vectorized environment

    To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.

    ```python
    >>> import gymnasium as gym
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
    >>> envs
    CartPoleVectorEnv(CartPole-v1, num_envs=3)
    >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    >>> envs
    SyncVectorEnv(CartPole-v1, num_envs=3)

    ```
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
            self,
            discrete_reward: bool,
            should_terminate: bool,
            step_time: float = 0.02,
            render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = step_time  # seconds between state updates
        self.kinematics_integrator = "euler"

        if discrete_reward:
            self._reward_fn = self._discrete_reward
        else:
            self._reward_fn = self._continuous_reward

        self._terminating_variant = should_terminate

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,  # TODO this is not true anymore
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        terminated = False   # This variant never terminates.

        if self.should_have_terminated:
            # We are in an absorbing state, no changes to the state or rewards.
            pass
        else:
            x, x_dot, theta, theta_dot = self.state
            force = self.force_mag if action == 1 else -self.force_mag
            costheta = math.cos(theta)
            sintheta = math.sin(theta)

            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (
                force + self.polemass_length * theta_dot**2 * sintheta
            ) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            if self.kinematics_integrator == "euler":
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta = theta + self.tau * theta_dot
                theta_dot = theta_dot + self.tau * thetaacc
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_dot = theta_dot + self.tau * thetaacc
                theta = theta + self.tau * theta_dot

            # Normalize theta to be between [-pi and pi].
            theta = (math.pi + theta) % (2*math.pi) - math.pi

            # x_bounded = min(self.x_threshold, max(x, -self.x_threshold))
            # We'll want to remove this, but for now we cannot.
            # if x != x_bounded:
            #     print("left the x bounds")
            # x = x_bounded

            self.state = (x, x_dot, theta, theta_dot)

            self.should_have_terminated = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )

        if self._terminating_variant:
            terminated = self.should_have_terminated

        if not terminated:
            pass
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1


        reward = self._reward_fn(self.state)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        start_state: Optional[Tuple[float, float, float, float]] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        if start_state is not None:
            if self.observation_space.contains(start_state):
                self.state = start_state
            else:
                raise ValueError(
                    f"Start state {start_state} is not within the "
                    "observation space {self.observation_space}."
                )
        else:
            low, high = utils.maybe_parse_reset_bounds(
                None, -0.05, 0.05  # default low
            )  # default high
            self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None
        self.should_have_terminated = False

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _discrete_reward(self, state):
        """The original."""
        if not self.should_have_terminated:
            return 1.0
        else:
            return 0.0

    def _continuous_reward(self, state):
        """A reward that slowly varies with the state."""
        # return math.exp(-20 * state[2]**2)
        # return ((np.cos(state[2])+1.0)/2.0)**4
        if not self.should_have_terminated:
            theta = state[2]
            if (theta < -self.theta_threshold_radians or
                theta > self.theta_threshold_radians):
                return 0.0
            else:
                return math.sqrt(
                    (self.theta_threshold_radians ** 2 - theta ** 2) 
                    / (self.theta_threshold_radians ** 2))
        else:
            return 0.0


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# class CartPoleVectorEnv(VectorEnv):
#     metadata = {
#         "render_modes": ["human", "rgb_array"],
#         "render_fps": 50,
#     }

#     def __init__(
#         self,
#         num_envs: int = 2,
#         max_episode_steps: int = 500,
#         render_mode: Optional[str] = None,
#     ):
#         self.num_envs = num_envs
#         self.max_episode_steps = max_episode_steps
#         self.render_mode = render_mode

#         self.gravity = 9.8
#         self.masscart = 1.0
#         self.masspole = 0.1
#         self.total_mass = self.masspole + self.masscart
#         self.length = 0.5  # actually half the pole's length
#         self.polemass_length = self.masspole * self.length
#         self.force_mag = 10.0
#         self.tau = 0.02  # seconds between state updates
#         self.kinematics_integrator = "euler"

#         self.steps = np.zeros(num_envs, dtype=np.int32)
#         self.prev_done = np.zeros(num_envs, dtype=np.bool_)

#         # Angle at which to fail the episode
#         self.theta_threshold_radians = 12 * 2 * math.pi / 360
#         self.x_threshold = 2.4

#         # Angle limit set to 2 * theta_threshold_radians so failing observation
#         # is still within bounds.
#         high = np.array(
#             [
#                 self.x_threshold * 2,
#                 np.finfo(np.float32).max,
#                 self.theta_threshold_radians * 2,
#                 np.finfo(np.float32).max,
#             ],
#             dtype=np.float32,
#         )

#         self.low = -0.05
#         self.high = 0.05

#         self.single_action_space = spaces.Discrete(2)
#         self.action_space = batch_space(self.single_action_space, num_envs)
#         self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
#         self.observation_space = batch_space(self.single_observation_space, num_envs)

#         self.screen_width = 600
#         self.screen_height = 400
#         self.screens = None
#         self.clocks = None
#         self.isopen = True
#         self.state = None

#         self.steps_beyond_terminated = None

#     def step(
#         self, action: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
#         assert self.action_space.contains(
#             action
#         ), f"{action!r} ({type(action)}) invalid"
#         assert self.state is not None, "Call reset before using step method."

#         x, x_dot, theta, theta_dot = self.state
#         force = np.sign(action - 0.5) * self.force_mag
#         costheta = np.cos(theta)
#         sintheta = np.sin(theta)

#         # For the interested reader:
#         # https://coneural.org/florian/papers/05_cart_pole.pdf
#         temp = (
#             force + self.polemass_length * theta_dot**2 * sintheta
#         ) / self.total_mass
#         thetaacc = (self.gravity * sintheta - costheta * temp) / (
#             self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
#         )
#         xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

#         if self.kinematics_integrator == "euler":
#             x = x + self.tau * x_dot
#             x_dot = x_dot + self.tau * xacc
#             theta = theta + self.tau * theta_dot
#             theta_dot = theta_dot + self.tau * thetaacc
#         else:  # semi-implicit euler
#             x_dot = x_dot + self.tau * xacc
#             x = x + self.tau * x_dot
#             theta_dot = theta_dot + self.tau * thetaacc
#             theta = theta + self.tau * theta_dot

#         self.state = np.stack((x, x_dot, theta, theta_dot))

#         terminated: np.ndarray = (
#             (x < -self.x_threshold)
#             | (x > self.x_threshold)
#             | (theta < -self.theta_threshold_radians)
#             | (theta > self.theta_threshold_radians)
#         )

#         self.steps += 1

#         truncated = self.steps >= self.max_episode_steps

#         reward = np.ones_like(terminated, dtype=np.float32)

#         # Reset all environments which terminated or were truncated in the last step
#         self.state[:, self.prev_done] = self.np_random.uniform(
#             low=self.low, high=self.high, size=(4, self.prev_done.sum())
#         )
#         self.steps[self.prev_done] = 0
#         reward[self.prev_done] = 0.0
#         terminated[self.prev_done] = False
#         truncated[self.prev_done] = False

#         self.prev_done = terminated | truncated

#         if self.render_mode == "human":
#             self.render()

#         return self.state.T.astype(np.float32), reward, terminated, truncated, {}

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[dict] = None,
#     ):
#         super().reset(seed=seed)
#         # Note that if you use custom reset bounds, it may lead to out-of-bound
#         # state/observations.
#         self.low, self.high = utils.maybe_parse_reset_bounds(
#             options, -0.05, 0.05  # default low
#         )  # default high
#         self.state = self.np_random.uniform(
#             low=self.low, high=self.high, size=(4, self.num_envs)
#         )
#         self.steps_beyond_terminated = None
#         self.steps = np.zeros(self.num_envs, dtype=np.int32)
#         self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

#         if self.render_mode == "human":
#             self.render()
#         return self.state.T.astype(np.float32), {}

#     def render(self):
#         if self.render_mode is None:
#             assert self.spec is not None
#             gym.logger.warn(
#                 "You are calling render method without specifying any render mode. "
#                 "You can specify the render_mode at initialization, "
#                 f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
#             )
#             return

#         try:
#             import pygame
#             from pygame import gfxdraw
#         except ImportError:
#             raise DependencyNotInstalled(
#                 "pygame is not installed, run `pip install gymnasium[classic_control]`"
#             )

#         if self.screens is None:
#             pygame.init()
#             if self.render_mode == "human":
#                 pygame.display.init()
#                 self.screens = [
#                     pygame.display.set_mode((self.screen_width, self.screen_height))
#                     for _ in range(self.num_envs)
#                 ]
#             else:  # mode == "rgb_array"
#                 self.screens = [
#                     pygame.Surface((self.screen_width, self.screen_height))
#                     for _ in range(self.num_envs)
#                 ]
#         if self.clocks is None:
#             self.clock = [pygame.time.Clock() for _ in range(self.num_envs)]

#         world_width = self.x_threshold * 2
#         scale = self.screen_width / world_width
#         polewidth = 10.0
#         polelen = scale * (2 * self.length)
#         cartwidth = 50.0
#         cartheight = 30.0

#         if self.state is None:
#             return None

#         for state, screen, clock in zip(self.state, self.screens, self.clocks):
#             x = self.state.T

#             self.surf = pygame.Surface((self.screen_width, self.screen_height))
#             self.surf.fill((255, 255, 255))

#             l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
#             axleoffset = cartheight / 4.0
#             cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
#             carty = 100  # TOP OF CART
#             cart_coords = [(l, b), (l, t), (r, t), (r, b)]
#             cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
#             gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
#             gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

#             l, r, t, b = (
#                 -polewidth / 2,
#                 polewidth / 2,
#                 polelen - polewidth / 2,
#                 -polewidth / 2,
#             )

#             pole_coords = []
#             for coord in [(l, b), (l, t), (r, t), (r, b)]:
#                 coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
#                 coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
#                 pole_coords.append(coord)
#             gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
#             gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

#             gfxdraw.aacircle(
#                 self.surf,
#                 int(cartx),
#                 int(carty + axleoffset),
#                 int(polewidth / 2),
#                 (129, 132, 203),
#             )
#             gfxdraw.filled_circle(
#                 self.surf,
#                 int(cartx),
#                 int(carty + axleoffset),
#                 int(polewidth / 2),
#                 (129, 132, 203),
#             )

#             gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

#             self.surf = pygame.transform.flip(self.surf, False, True)
#             screen.blit(self.surf, (0, 0))

#         if self.render_mode == "human":
#             pygame.event.pump()
#             [clock.tick(self.metadata["render_fps"]) for clock in self.clocks]
#             pygame.display.flip()

#         elif self.render_mode == "rgb_array":
#             return [
#                 np.transpose(
#                     np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
#                 )
#                 for screen in self.screens
#             ]

#     def close(self):
#         if self.screens is not None:
#             import pygame

#             pygame.display.quit()
#             pygame.quit()
#             self.isopen = False