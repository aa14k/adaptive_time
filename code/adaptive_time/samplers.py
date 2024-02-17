import numpy as np


class UniformSampler:
    def __init__(self, num_steps: int, spacing: int):
        self.num_steps = num_steps
        self.spacing = spacing

    def sample_time(self):
        return np.arange(0, self.num_steps, self.spacing)
