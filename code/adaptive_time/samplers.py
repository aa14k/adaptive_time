from abc import ABC, abstractclassmethod
from typing import Any

import numpy as np


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


class QuadratureSampler(Sampler):
    def __init__(self, num_steps: int, tolerance: float):
        self.num_steps = num_steps
        self.tolerance = tolerance
        self._sample_times = [0, 1]

    def adapt(self, traj: Any):
        pass

    def sample_time(self):
        return np.array(self._sample_times)
