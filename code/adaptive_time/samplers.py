from abc import ABC, abstractclassmethod
from typing import Any, List

import numpy as np

from adaptive_time import utils


class Sampler2(ABC):
    """Samplers return a list of indices to do updates at."""

    def pivots(self, trajectory) -> np.ndarray:
        """Returns an array of indices to do updates at."""
        pass


class AdaptiveQuadratureSampler2(Sampler2):
    """Identifies pivots using quadrature methods."""

    def __init__(self, tolerance: float) -> None:
        super().__init__()
        self._tolerance = tolerance
    
    def pivots(self, trajectory) -> np.ndarray:
        N = len(trajectory)
        rewards = np.zeros((2, N))
        for idx, traj in enumerate(trajectory):
            rewards[0, idx] = traj[2]
            rewards[1, idx] = idx
        idxes = {}
        _ = utils.approx_integrate(rewards, self._tolerance, idxes)
        pivots = list(idxes.keys())
        return np.array(pivots, dtype=np.int32)


class UniformSampler2(Sampler2):
    """Returns uniformly spaced pivots."""
    def __init__(self, spacing: int) -> None:
        super().__init__()
        self._spacing = spacing

    def pivots(self, trajectory) -> np.ndarray:
        return np.arange(0, len(trajectory), self._spacing, np.int32)


class Sampler(ABC):
    def adapt(self, **kwargs):
        pass

    @abstractclassmethod
    def sample_time(self):
        pass


class UniformSampler(Sampler):
    def __init__(self, num_steps: int, spacing: int):
        self.num_steps = num_steps
        self.spacing = spacing

    def sample_time(self):
        return np.arange(0, self.num_steps, self.spacing)


class AdaptiveQuadratureSampler(Sampler):
    def __init__(
        self,
        dt: float,
        num_steps: int,
        tolerance_init: float,
        integral_rule: str = "trapezoid",
        update_when_best: bool = True,
    ):
        #self.dt = dt
        self.dt = 1
        self.num_steps = num_steps
        self.tolerance_init = tolerance_init
        self._sample_times = np.arange(num_steps)
        self.best_rew = -np.inf
        self.update_when_best = update_when_best

        if integral_rule == "trapezoid":
            self.integral_rule = self._trapezoid_rule
        else:
            raise NotImplementedError

    def _trapezoid_rule(self, rews: List[float]):
        if not len(rews):
            return 0.0
        return 0.5 * len(rews) * self.dt * (rews[0] + rews[-1])

    def _adapt(
        self,
        rews: List[float],
        t_start: int,
        t_end: int,
        curr_seg: float,
        tolerance: float,
    ):
        if t_end <= t_start:
            return ([], 0, 1)

        t_mid = int(np.floor(0.5 * (t_start + t_end)))
        left_seg = self.integral_rule(rews[t_start:t_mid])
        right_seg = self.integral_rule(rews[t_mid:t_end])

        total_seg = left_seg + right_seg
        if np.abs(curr_seg - total_seg) < tolerance:
            return ([t_mid], total_seg, 1)

        next_tolerance = 0.5 * tolerance
        left_sample_times, left_seg, left_calls = self._adapt(
            rews, t_start, t_mid, left_seg, next_tolerance
        )
        right_sample_times, right_seg, right_calls = self._adapt(
            rews, t_mid, t_end, right_seg, next_tolerance
        )

        sample_times = [t_mid]
        sample_times.extend(left_sample_times)
        sample_times.extend(right_sample_times)
        sample_times = np.unique(sample_times)

        return (sample_times, left_seg + right_seg, left_calls + right_calls + 1)

    def adapt(self, trajs: Any):
        rews = np.mean([traj["rews"] for traj in trajs], axis=0)
        curr_seg = self.integral_rule(rews)
        sample_times, total_seg, num_calls = self._adapt(
            rews, 0, self.num_steps + 1, curr_seg, self.tolerance_init
        )
        if self.update_when_best:
            total_reward = np.sum(rews)
            if self.best_rew <= total_reward:
                self._sample_times = np.concatenate(([0], sample_times)).astype(int)
                self.best_rew = total_reward
            else:
                self._sample_times = np.sort(np.random.permutation(self.num_steps)[:20])
        else:
            self._sample_times = np.concatenate(([0], sample_times)).astype(int)
        return sample_times, total_seg, num_calls

    def sample_time(self):
        return np.array(self._sample_times)
