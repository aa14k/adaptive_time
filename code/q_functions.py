from abc import ABC, abstractclassmethod
from types import SimpleNamespace
from typing import Any

from features import MountainCarTileCoder

import numpy as np

class QFunction(ABC):
    @abstractclassmethod
    def update(self, trajs: Any):
        pass

    @abstractclassmethod
    def greedy_action(self, obs: Any):
        pass


class MountainCarTileCodingQ(QFunction):
    def __init__(self, agent_config: SimpleNamespace) -> None:
        super().__init__()
        self.agent_config = agent_config
        self.tile_coder = MountainCarTileCoder(
            iht_size=getattr(agent_config, "iht_size", 4096),
            num_tilings=getattr(agent_config, "num_tilings", 8),
            num_tiles=getattr(agent_config, "num_tiles", 8)
        )
        rng = np.random.RandomState(agent_config.seed)
        self.action_space = agent_config.action_space
        self.parameters = rng.randn(
            self.tile_coder.num_tilings,
            len(agent_config.action_space),
        )

    def update(self, trajs: Any):
        pass

    def greedy_action(self, obs: Any):
        feature = self.tile_coder.get_tiles(*obs)
        q_vals = feature @ self.parameters
        return self.action_space[np.argmax(q_vals)]
