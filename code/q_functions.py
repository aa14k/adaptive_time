from abc import ABC, abstractclassmethod
from types import SimpleNamespace
from typing import Any, List, Dict

from code.features import MountainCarTileCoder
from code.utils import softmax

import numpy as np


class QFunction(ABC):
    def update(self, **kwargs) -> Dict[str, Any]:
        pass

    @abstractclassmethod
    def greedy_action(self, obs: Any, **kwargs):
        pass

    @abstractclassmethod
    def sample_action(self, obs: Any, **kwargs):
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
        print(agent_config)
        rng = np.random.RandomState(agent_config.seed)
        self.action_space = agent_config.action_space
        self.parameters = rng.randn(
            self.tile_coder.num_tilings,
            len(agent_config.action_space),
        ) * getattr(agent_config, "param_init_std", 0.1) + getattr(
            agent_config, "param_init_mean", 0.0
        )

        if agent_config.update_rule == "monte_carlo":
            self.update = self.mc_update
        elif agent_config.update_rule == "sarsa":
            self.update = self.sarsa_update
        else:
            raise NotImplementedError

    def sarsa_update(
        self,
        curr_tx: Any,
        next_tx: Any,
        curr_observe_sample: int,
        next_observe_sample: int,
        max_time: int,
    ):
        curr_feature = self.tile_coder.get_tiles(*curr_tx["obs"])
        q_val = (curr_feature @ self.parameters)[curr_tx["act"]]

        next_q_val = 0.0
        if not next_tx["done"]:
            next_q_val = (self.tile_coder.get_tiles(*next_tx["obs"]) @ self.parameters)[
                next_tx["act"]
            ]

        target = (min(next_observe_sample, max_time) - curr_observe_sample) * curr_tx[
            "rew"
        ] + next_q_val
        td_error = target - q_val
        update = td_error * curr_feature

        self.parameters[:, curr_tx["act"]] = (
            self.parameters[:, curr_tx["act"]]
            + self.agent_config.learning_rate * update
        )

        return dict(
            target=target,
            td_error=td_error,
            q_val=q_val,
            next_q_val=next_q_val,
            rew=curr_tx["rew"],
            act=curr_tx["act"],
            param_norms=np.linalg.norm(self.parameters, axis=0),
            update_norm=np.linalg.norm(update),
        )

    def mc_update(
        self,
        disc_trajs: Any,
        ep_horizons: Any,
        observe_times: Any,
        max_time: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get dict of ndarrays, which results in:
        {
            "obss": ndarray,
            "acts": ndarray,
            "rews": ndarray,
        }
        We assume that the arrays are padded such that the horizon is consistent.
        """
        disc_trajs = {
            k: np.array([traj[k] for traj in disc_trajs]) for k in disc_trajs[0]
        }

        # Compute Monte Carlo return
        rets = compute_disc_mc_return(disc_trajs["rews"], observe_times, max_time)

        # Get tile-coding features
        features = np.array(
            [
                [self.tile_coder.get_tiles(*obs) for obs in obss]
                for obss in disc_trajs["obss"]
            ]
        )

        # Compute Q-values and TD errors
        q_vals = features @ self.parameters
        q_vals_act = np.take_along_axis(q_vals, disc_trajs["acts"][..., None], axis=-1)
        ep_mask = (1 - np.eye(max_time + 1)[np.array(ep_horizons) + 1])[..., :-1][
            :, observe_times
        ].reshape(-1, 1, 1)

        td_error = rets[..., None] - q_vals_act
        acts_one_hot = np.eye(len(self.action_space))[disc_trajs["acts"]].reshape(
            -1, len(self.action_space), 1
        )

        """
        This is a little bit trickier...
        We need to:
        1. Use ep_mask to cut off transitions after termination
        2. Use acts_one_hot to only look at Q-values of taken actions
        """
        per_sample_update = (
            np.tile(
                (td_error * features).reshape(-1, self.tile_coder.num_tilings)[:, None],
                reps=(1, len(self.action_space), 1),
            )
            * acts_one_hot
            * ep_mask
        )

        """
        Make sure the average considers the actions taken and actual horizon of the trajectories.
        Also, make sure that actions that are never taken is not updated (through enforcing 0s).
        """
        average_update = (
            np.sum(per_sample_update, axis=0) / np.sum(acts_one_hot * ep_mask, axis=0)
        ).T
        average_update[np.where(1 - np.isfinite(average_update))] = 0
        self.parameters = (
            self.parameters + self.agent_config.learning_rate * average_update
        )

        return dict(
            mean_td_error=np.mean(td_error**2),
            mean_q_vals=np.mean(q_vals, axis=(0, 1)),
            param_norms=np.linalg.norm(self.parameters, axis=0),
            update_norm=np.linalg.norm(average_update, axis=0),
            returns=np.mean(rets[:, 0]),
            action_frequency=(
                np.sum(acts_one_hot * ep_mask, axis=0) / np.sum(ep_mask, axis=0)
            ).T,
        )

    def greedy_action(self, obs: Any, **kwargs):
        feature = self.tile_coder.get_tiles(*obs)
        q_vals = feature @ self.parameters
        return np.argmax(q_vals)

    def sample_action(self, obs: Any, **kwargs):
        feature = self.tile_coder.get_tiles(*obs)
        q_vals = feature @ self.parameters
        probs = softmax(q_vals)
        return np.random.choice(len(self.action_space), p=probs)
