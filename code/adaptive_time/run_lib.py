"""USE run_control.py or run_eval.py.

Implements the main programs for the adaptive time project.
"""

import argparse
import copy
import pickle
from datetime import datetime
import enum
import random

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import adaptive_time.utils
from adaptive_time.features import Fourier_Features
from adaptive_time import environments
from adaptive_time import mc2



parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name",
    type=str,
    required=True,
    help="The name for the experiment. This will be used to name the output directory.",
)


class BudgetType(enum.Enum):
    INTERACTIONS = 1
    UPDATES = 2


_NUM_ACTIONS = 2


def register_gym_envs():
    gym.register(
        id="CartPole-OURS-v0",
        entry_point="adaptive_time.environments.cartpole:CartPoleEnv",
        vector_entry_point="adaptive_time.environments.cartpole:CartPoleVectorEnv",
        max_episode_steps=500,
        reward_threshold=475.0,
    )


def reset_randomness(seed, env):
    """Resets all randomness, except for the environment's seed."""
    random.seed(seed)
    np.random.seed(seed)
    # env.seed(seed)
    env.action_space.seed(seed)    


class ActionIterator:
    def __init__(self, action_sequence):
        self.action_sequence = action_sequence
        self.idx = 0
    
    def __call__(self, state):
        action = self.action_sequence[self.idx]
        self.idx += 1
        return action


def run_experiment(
        seed, env, phi, sampler, epsilon,
        budget, budget_type: BudgetType,
        termination_prob, max_env_steps, do_weighing,
        gamma, file_postfix, tqdm=None, print_trajectory=False,
        save_threshold=None,
        weights_to_evaluate=None, policy_to_evaluate=None):
    """Keeps interacting until the budget is (approximately) used up, returns stats.
    
    Note that the budgets are in terms of processed interactions (or updates). We
    will do one last episode, even if the budget is used up, so that we can evaluate
    the final weights.
    """
    if tqdm is None:
        tqdm_use = lambda x: x

    if max_env_steps is not None:
        raise ValueError("max_env_steps is not supported in this version.")

    # We record:
    returns = []  # The total return at the end of each episode.
    # And how many updates/interactions we have done by that point:
    total_pivots = [0]
    total_interactions = [0]
    num_episode = [0]

    # Each of the following will record the values for both actions.
    # Each element of these will be a (2,)-np.array, so we can just stack them.
    returns_per_episode_q = []
    predicted_returns_q = []
    # The value.
    returns_per_episode_v = []
    predicted_returns_v = []

    reset_randomness(seed, env)

    observation, _ = env.reset(seed=seed)
    d = len(phi.get_fourier_feature(observation))
    assert d == phi.num_parameters
    features = np.identity(2 * d)   # An estimate of A = xx^T
    targets = np.zeros(2 * d)  # An estimate of b = xG
    weights = np.zeros(2 * d)   # The weights that approximate A^{-1} b

    x_0 = phi.get_fourier_feature([0,0,0,0])  # the initial state
    x_sa0 = mc2.phi_sa(x_0, 0)
    x_sa1 = mc2.phi_sa(x_0, 1)

    def remaining_steps():
        if budget_type == BudgetType.INTERACTIONS:
            return budget - total_interactions[-1]
        elif budget_type == BudgetType.UPDATES:
            return budget - total_pivots[-1]
        else:
            raise ValueError("Unknown budget type")

    def policy(state, weights):
        """Returns the action to take, and maybe the prob of all actions"""
        if random.random() < epsilon:
            action = env.action_space.sample()
            return action

        # Otherwise calculate the best action.
        x = phi.get_fourier_feature(state)
        qs = np.zeros(_NUM_ACTIONS)
        for action in range(_NUM_ACTIONS):
            x_sa = mc2.phi_sa(x, action)
            qs[action] = np.inner(x_sa.flatten(), weights)
        # adaptive_time.utils.softmax(qs, 1)
        
        return adaptive_time.utils.argmax(qs)
    
    if (weights_to_evaluate is not None
        and policy_to_evaluate is not None):
        raise ValueError(
            "Both weights_to_evaluate and policy_to_evaluate are set.")

    control = None         # Whether we are doing control.
    maybe_switch_policy = False
    if weights_to_evaluate is not None:
        # Evaluate the given weights, not doing control.
        policy_to_use = lambda s: policy(state=s, weights=weights_to_evaluate)
        control = False
    elif policy_to_evaluate is not None:
        # Evaluate the given sequence of actions, not doing control.
        maybe_switch_policy = True
        control = False
    else:
        policy_to_use = lambda s: policy(state=s, weights=weights)
        control = True

    # For evaluation, we keep track of how many times we took
    # each action in the first timestep.
    # SHOULD we do this each time we visit the initial state?
    first_action_counts = np.zeros(_NUM_ACTIONS)

    all_weights = []
    first_actions = []
    while remaining_steps() > 0:
        if maybe_switch_policy:
            if random.random() < 0.5:
                policy_to_use = ActionIterator(policy_to_evaluate[0])
            else:
                policy_to_use = ActionIterator(policy_to_evaluate[1])

        trajectory, early_term = environments.generate_trajectory(
                env, policy=policy_to_use,
                # env, policy=lambda s: policy(state=s, weights=weights),
                termination_prob=termination_prob, max_steps=max_env_steps)
        
        # Process and record the return.
        returns.append(sum(ts[2] for ts in trajectory))
        if save_threshold is not None and returns[-1] >= save_threshold:
            np.save(f"cartpole_weights_{file_postfix}_ret{returns[-1]}.npy", weights)

        if print_trajectory:
            print("trajectory-len: ", len(trajectory), "; trajectory:")
            for idx, (o, a, r, o_) in enumerate(trajectory):
                # * ignore reward, as it is always the same here.
                # * o_ is the same as the next o.
                print(f"* {idx:4d}: o: {o}\n\t --> action: {a}")

        assert early_term is False, "We should not terminate early in this experiment."

        # Do updates, record stats from the processed trajectory.
        (weights, targets, features, cur_avr_returns_q, cur_avr_returns_v,
         num_pivots) = mc2.ols_monte_carlo(
            trajectory, sampler, tqdm_use, phi, weights, targets,
            features, x_0, do_weighing, gamma)
        
        # Update the stats.
        total_pivots.append(total_pivots[-1] + num_pivots)
        total_interactions.append(total_interactions[-1] + len(trajectory))
        num_episode.append(num_episode[-1] + 1)
        all_weights.append(weights)
        
        # Store the empirical and predicted returns. For any episode, we may
        # or may not have empirical returns for both actions. When we don't have an
        # estimate, `nan` is returned.
        returns_per_episode_q.append(cur_avr_returns_q)
        predicted_returns_q.append(np.array(
            [np.inner(x_sa0.flatten(), weights),
                np.inner(x_sa1.flatten(), weights)]))
        
        # Also update the value estimates.
        returns_per_episode_v.append(cur_avr_returns_v)
        if control:
            predicted_returns_v.append(
                adaptive_time.utils.v_from_eps_greedy_q(
                    epsilon, predicted_returns_q[-1]))
        else:
            first_action = trajectory[0][1]
            first_action_counts[first_action] += 1
            assert np.sum(first_action_counts) == num_episode[-1]
            v_est = np.dot(
                first_action_counts, predicted_returns_q[-1]) / num_episode[-1]
            predicted_returns_v.append(v_est)
            first_actions.append(first_action)
    
    # The following variant produces plots where we can see
    # the effect of the last update.
    # Do one more evaluation run.
    if not maybe_switch_policy:
        trajectory, early_term = environments.generate_trajectory(
            env, policy=policy_to_use,
            # env, policy=lambda s: policy(state=s, weights=weights),
            termination_prob=termination_prob, max_steps=max_env_steps)
        returns.append(sum(ts[2] for ts in trajectory))

    return {
        "total_return": returns,
        "total_pivots": total_pivots,
        "total_interactions": total_interactions,
        "num_episode": num_episode,
        "returns_per_episode_q": returns_per_episode_q,
        "predicted_returns_q": predicted_returns_q,
        "returns_per_episode_v": returns_per_episode_v,
        "predicted_returns_v": predicted_returns_v,
        "all_weights": all_weights,
        "first_actions": first_actions,
    }
    
    # The following variant produces plots where we
    # ensure that the last x-points are within the budget.
    # Do one more evaluation run.

    # return {
    #     "total_return": total_return,
    #     "total_pivots": total_pivots[:-1],
    #     "total_interactions": total_interactions[:-1],
    #     "num_episode": num_episode[:-1],
    #     "returns_per_episode_q": returns_per_episode_q,
    #     "predicted_returns_q": predicted_returns_q,
    # }
        



#     ============      Configure Features       ============
def make_features():
    phi = Fourier_Features()
    phi.init_fourier_features(4,4)
    x_thres = 4.8
    theta_thres = 0.418
    phi.init_state_normalizers(
        np.array([x_thres,2.0,theta_thres,1]),
        np.array([-x_thres,-2.0,-theta_thres,-1]))
    return phi


def run_generic(config_dict, samplers_tried):
    args = parser.parse_args()

    date_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = adaptive_time.utils.slugify(args.exp_name)

    adaptive_time.utils.set_directory_in_project()

    register_gym_envs()
    env = gym.make('CartPole-OURS-v0')

    orig_config = copy.deepcopy(config_dict)

    seed = config_dict.pop("seed")
    save_limit = config_dict.pop("save_limit")
    termination_prob = config_dict.pop("termination_prob")
    max_env_steps = config_dict.pop("max_env_steps")
    epsilon = config_dict.pop("epsilon")
    gamma = config_dict.pop("gamma")
    budget = config_dict.pop("budget")
    budget_type = config_dict.pop("budget_type")
    num_runs = config_dict.pop("num_runs")
    tau = config_dict.pop("tau")
    weights_to_evaluate = config_dict.pop("weights_to_evaluate")
    policy_to_evaluate = config_dict.pop("policy_to_evaluate")
    do_weighing = config_dict.pop("do_weighing")
    if config_dict:
        raise ValueError(f"Unknown additional configs:\n{config_dict}")
    
    env.stepTime(tau)
    phi = make_features()

    if weights_to_evaluate is not None:
        if weights_to_evaluate == 0:
            # Set the weights to evaluate to 0 of the right size.
            weights_to_evaluate = np.zeros(phi.num_parameters * 2)
        elif isinstance(weights_to_evaluate, str):
            # Load the weights from the given file.
            weights_to_evaluate = np.load(weights_to_evaluate)
    if policy_to_evaluate is not None:
        assert len(policy_to_evaluate) == 3
        if isinstance(policy_to_evaluate[0], str):
            # Load the policy from the given file.
            pol1 = np.load(adaptive_time.utils.get_abs_path(
                policy_to_evaluate[0]))
        else:
            raise ValueError()
        if isinstance(policy_to_evaluate[1], str):
            # Load the policy from the given file.
            pol2 = np.load(adaptive_time.utils.get_abs_path(
                policy_to_evaluate[1]))
        else:
            raise ValueError()
        policy_to_evaluate = (pol1, pol2, policy_to_evaluate[2])

    results = {}
    for name, sampler in tqdm(samplers_tried.items()):
        print(name, sampler)
        #results[name] = []
        results[name] = Parallel(n_jobs = num_runs)(
            delayed(run_experiment)(
                seed+run, env, phi, sampler, epsilon, budget, budget_type,
                termination_prob, max_env_steps, do_weighing=do_weighing,
                gamma=gamma,
                file_postfix=name + "_" + date_string,
                tqdm=None, print_trajectory=False, save_threshold=save_limit,
                weights_to_evaluate=weights_to_evaluate,
                policy_to_evaluate=policy_to_evaluate)
                for run in range(num_runs)
            )

    print()
    print("DONE!")

    adaptive_time.utils.set_directory_in_project(
        f"exp_results/{date_string}_{exp_name}",
        create_dirs=True)

    filename = f"exp_data.pkl"
    print(f"Saving results to {filename}...")
    with open(filename, "wb") as f:
        pickle.dump({"results": results, "config": orig_config}, f)
    print("Saved results.")

    adaptive_time.utils.set_directory_in_project()


