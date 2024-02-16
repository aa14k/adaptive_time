from abc import ABC, abstractclassmethod
from types import SimpleNamespace
from typing import Any, List, Dict

from features import MountainCarTileCoder

import numpy as np


class QFunction(ABC):
    @abstractclassmethod
    def update(
        self, trajs: Any, ep_horizons: Any, observe_times: Any, max_time: Any, **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractclassmethod
    def greedy_action(self, obs: Any, **kwargs):
        pass


def compute_disc_mc_return(
    rewards: np.ndarray, observe_times: List[int], max_time: int
):
    """
    Compute the discretized Monte Carlo return
    - rewards (np.ndarray): A (N x H) matrix storing the H rewards per each of N episodes
    - observe_times (List[int]): The observe indices that resulted in the rewards
    - max_time (int): The maximum horizon length
    """
    (num_trajs, traj_len) = rewards.shape
    # Assumes that rewards have same length
    mc_returns = np.zeros(shape=(num_trajs, traj_len + 1))
    padded_time = list(observe_times) + [max_time]
    for timestep_i in range(traj_len - 1, -1, -1):
        mc_returns[:, timestep_i] += (
            padded_time[timestep_i + 1] - padded_time[timestep_i]
        ) * rewards[:, timestep_i] + mc_returns[:, timestep_i + 1]
    return mc_returns[:, :-1]


class MountainCarTileCodingQ(QFunction):
    def __init__(self, agent_config: SimpleNamespace) -> None:
        super().__init__()
        self.agent_config = agent_config
        self.tile_coder = MountainCarTileCoder(
            iht_size=getattr(agent_config, "iht_size", 4096),
            num_tilings=getattr(agent_config, "num_tilings", 8),
            num_tiles=getattr(agent_config, "num_tiles", 8),
        )
        rng = np.random.RandomState(agent_config.seed)
        self.action_space = agent_config.action_space
        self.parameters = rng.randn(
            self.tile_coder.num_tilings,
            len(agent_config.action_space),
        )

    def update(
        self, trajs: Any, ep_horizons: Any, observe_times: Any, max_time: Any, **kwargs
    ) -> Dict[str, Any]:
        trajs = {k: np.array([traj[k] for traj in trajs]) for k in trajs[0]}
        rets = compute_disc_mc_return(trajs["rews"], observe_times, max_time)
        features = np.array(
            [
                [self.tile_coder.get_tiles(*obs) for obs in obss]
                for obss in trajs["obss"]
            ]
        )
        q_vals = features @ self.parameters
        q_vals_act = np.take_along_axis(q_vals, trajs["acts"][..., None], axis=-1)
        ep_mask = (1 - np.eye(len(observe_times) + 1)[np.array(ep_horizons) + 1])[
            ..., :-1
        ].reshape(-1, 1, 1)

        td_error = rets[..., None] - q_vals_act
        acts_one_hot = np.eye(len(self.action_space))[trajs["acts"]].reshape(
            -1, len(self.action_space), 1
        )
        per_sample_update = (
            np.tile(
                (td_error * features).reshape(-1, self.tile_coder.num_tilings)[:, None],
                reps=(1, len(self.action_space), 1),
            )
            * acts_one_hot
            * ep_mask
        )
        average_update = (
            np.sum(per_sample_update, axis=0) / np.sum(acts_one_hot * ep_mask, axis=0)
        ).T
        average_update[np.where(1 - np.isfinite(average_update))] = 0
        self.parameters = (
            self.parameters + self.agent_config.learning_rate * average_update
        )

        return dict(
            mean_td_error=np.mean(td_error ** 2),
            param_norms=np.linalg.norm(self.parameters, axis=0),
            update_norm=np.linalg.norm(average_update, axis=0),
            returns=np.mean(rets[:, 0]),
            action_frequency=np.mean(acts_one_hot, axis=0).T,
        )

    def greedy_action(self, obs: Any, **kwargs):
        feature = self.tile_coder.get_tiles(*obs)
        q_vals = feature @ self.parameters
        return np.argmax(q_vals)
