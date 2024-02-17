from abc import ABC, abstractclassmethod
from types import SimpleNamespace
from typing import Any, List, Dict

from adaptive_time.features import MountainCarTileCoder
from adaptive_time.utils import softmax, argmax

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
    rewards: np.ndarray,
    observe_times: List[int],
    max_time: int,
    dt_sec: float,
):
    """
    Compute the discretized Monte Carlo return
    - rewards (np.ndarray): A (N x H) matrix storing the H rewards per each of N episodes
    - observe_times (List[int]): The observe indices that resulted in the rewards
    - max_time (int): The maximum horizon length
    - dt_sec (float): The most fine-grain delta time
    """
    (num_trajs, traj_len) = rewards.shape
    # Assumes that rewards have same length
    mc_returns = np.zeros(shape=(num_trajs, traj_len + 1))
    padded_time = list(observe_times) + [max_time]
    for timestep_i in range(traj_len - 1, -1, -1):
        mc_returns[:, timestep_i] += (
            padded_time[timestep_i + 1] - padded_time[timestep_i]
        ) * dt_sec * rewards[:, timestep_i] + mc_returns[:, timestep_i + 1]
    return mc_returns[:, :-1]


class MountainCarTileCodingQ(QFunction):
    def __init__(self, agent_config: SimpleNamespace) -> None:
        super().__init__()
        self.agent_config = agent_config
        self.num_parameters = getattr(agent_config, "iht_size", 4096)
        self.tile_coder = MountainCarTileCoder(
            iht_size=self.num_parameters,
            num_tilings=getattr(agent_config, "num_tilings", 8),
            num_tiles=getattr(agent_config, "num_tiles", 8),
        )
        rng = np.random.RandomState(agent_config.seed)
        self.action_space = agent_config.action_space
        self.parameters = rng.randn(
            self.num_parameters,
            len(agent_config.action_space),
        ) * getattr(agent_config, "param_init_std", 0.1) + getattr(
            agent_config, "param_init_mean", 0.0
        )

        if agent_config.update_rule == "monte_carlo":
            self.update = self.mc_update
        elif agent_config.update_rule == "batched_monte_carlo":
            self.update = self.batched_mc_update
        elif agent_config.update_rule == "sarsa":
            self.update = self.sarsa_update
        else:
            raise NotImplementedError

    def get_feature(self, obs: Any) -> np.ndarray:
        active_tiles = self.tile_coder.get_tiles(*obs)
        feature = np.zeros(self.num_parameters)
        feature[active_tiles] = 1
        return feature

    def sarsa_update(
        self,
        curr_tx: Any,
        next_tx: Any,
        curr_observe_sample: int,
        next_observe_sample: int,
        max_time: int,
        dt_sec: float,
        **kwargs,
    ):
        curr_feature = self.get_feature(curr_tx["curr_obs"])
        q_val = (curr_feature @ self.parameters)[curr_tx["act"]]

        next_q_val = 0.0
        if not next_tx["done"]:
            next_q_val = (self.get_feature(next_tx["curr_obs"]) @ self.parameters)[
                next_tx["act"]
            ]

        target = (
            min(next_observe_sample, max_time) - curr_observe_sample
        ) * dt_sec * curr_tx["rew"] + next_q_val
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
            curr_tx=curr_tx,
            next_tx=next_tx,
            param_norms=np.linalg.norm(self.parameters, axis=0),
            update_norm=np.linalg.norm(update),
        )

    def mc_update(
        self,
        disc_trajs: Any,
        ep_horizons: Any,
        observe_times: Any,
        max_time: Any,
        dt_sec: float,
        **kwargs,
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
        rets = compute_disc_mc_return(
            disc_trajs["rews"], observe_times, max_time - 1, dt_sec
        )

        # Construct mask to remove finished timesteps
        ep_mask = (1 - np.eye(max_time + 1)[np.array(ep_horizons) + 1])[..., :-1]

        average_updates = []
        action_frequencies = np.zeros(len(self.action_space))

        for timestep in observe_times[::-1]:
            # Get action a_t, feature x_t and return G_t
            curr_acts = disc_trajs["acts"][:, timestep]
            curr_features = np.array(
                [self.get_feature(obs) for obs in disc_trajs["obss"][:, timestep]]
            )
            curr_rets = rets[:, timestep]

            # Compute Q-values
            q_vals = curr_features @ self.parameters
            q_vals_act = np.take_along_axis(q_vals, curr_acts[..., None], axis=-1)

            # Compute TD error
            td_error = curr_rets[..., None] - q_vals_act

            # Perform update only to paraemters with actions taken, and with unfinished episodes
            acts_one_hot = np.eye(len(self.action_space))[curr_acts].reshape(
                -1, len(self.action_space), 1
            )
            include_timestep_mask = ep_mask[:, timestep]

            per_sample_update = (
                np.tile(
                    (td_error * curr_features).reshape(-1, 1, self.num_parameters),
                    reps=(1, len(self.action_space), 1),
                )
                * acts_one_hot
                * include_timestep_mask
            )

            # Take average across episode---deal with actions that are not taken
            average_update = (
                np.sum(per_sample_update, axis=0)
                / np.sum(acts_one_hot * include_timestep_mask, axis=0)
            ).T
            average_update[np.where(1 - np.isfinite(average_update))] = 0

            self.parameters = (
                self.parameters + self.agent_config.learning_rate * average_update
            )

            average_updates.append(average_update)
            action_frequencies += np.sum(
                acts_one_hot * include_timestep_mask, axis=0
            ).flatten()

        return dict(
            mean_td_error=np.mean(td_error**2),
            mean_q_vals=np.mean(q_vals, axis=(0, 1)),
            param_norms=np.linalg.norm(self.parameters, axis=0),
            update_norm=np.linalg.norm(np.mean(average_updates, axis=0), axis=0),
            returns=np.mean(rets[:, 0]),
            action_frequency=action_frequencies / np.sum(action_frequencies),
        )

    def batched_mc_update(
        self,
        disc_trajs: Any,
        ep_horizons: Any,
        observe_times: Any,
        max_time: Any,
        dt_sec: float,
        **kwargs,
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

        # print(ep_horizons)
        if ep_horizons[0] < 199:
            import ipdb

            ipdb.set_trace()

        # Compute Monte Carlo return
        rets = compute_disc_mc_return(
            disc_trajs["rews"], observe_times, max_time - 1, dt_sec
        )

        # Get tile-coding features
        features = np.array(
            [[self.get_feature(obs) for obs in obss] for obss in disc_trajs["obss"]]
        )

        # Compute Q-values and TD errors
        q_vals = features @ self.parameters
        q_vals_act = np.take_along_axis(q_vals, disc_trajs["acts"][..., None], axis=-1)
        ep_mask = (1 - np.eye(max_time + 1)[np.array(ep_horizons) + 1])[..., :-1][
            :, observe_times
        ].reshape(-1, 1, 1)

        # import ipdb
        # ipdb.set_trace()

        # print(rets, len(rets[0]))
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
                (td_error * features).reshape(-1, self.num_parameters)[:, None],
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
        feature = self.get_feature(obs)
        q_vals = feature @ self.parameters
        return argmax(q_vals)

    def sample_action(self, obs: Any, temperature: float = 1, **kwargs):
        feature = self.get_feature(obs)
        q_vals = feature @ self.parameters
        probs = softmax(q_vals / temperature)
        return np.random.choice(len(self.action_space), p=probs)
