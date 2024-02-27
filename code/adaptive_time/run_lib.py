"""USE run_control.py or run_eval.py.

Implements the main programs for the adaptive time project.
"""


import gymnasium as gym
from adaptive_time.features import Fourier_Features
import numpy as np
from tqdm import tqdm

import random
from joblib import Parallel, delayed

import adaptive_time.utils
from adaptive_time import environments
from adaptive_time import mc2

import pickle
from datetime import datetime

import copy
import enum


class BudgetType(enum.Enum):
    INTERACTIONS = 1
    UPDATES = 2



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


def run_experiment(
        seed, env, phi, sampler, epsilon,
        budget, budget_type: BudgetType,
        termination_prob, max_env_steps,
        gamma, file_postfix, tqdm=None, print_trajectory=False,
        save_threshold=None,
        weights_to_evaluate=None):
    """Keeps interacting until the budget is (approximately) used up, returns stats.
    
    Note that the budgets are in terms of processed interactions (or updates). We
    will do one last episode, even if the budget is used up, so that we can evaluate
    the final weights.
    """
    if tqdm is None:
        tqdm_use = lambda x: x

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
        if random.random() < epsilon:
            return env.action_space.sample()
        # Otherwise calculate the best action.
        x = phi.get_fourier_feature(state)
        qs = np.zeros(2)
        for action in [0, 1]:
            x_sa = mc2.phi_sa(x, action)
            qs[action] = np.inner(x_sa.flatten(), weights)
        # adaptive_time.utils.softmax(qs, 1)
        return adaptive_time.utils.argmax(qs)
    
    if weights_to_evaluate is not None:
        # Evaluate the given weights, not doing control.
        policy_to_use = lambda s: policy(state=s, weights=weights_to_evaluate)
    else:
        policy_to_use = lambda s: policy(state=s, weights=weights)


    while remaining_steps() > 0:

        trajectory, early_term = environments.generate_trajectory(
                env, policy=policy_to_use,
                # env, policy=lambda s: policy(state=s, weights=weights),
                termination_prob=termination_prob, max_env_steps=max_env_steps)
        
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
        weights, targets, features, cur_avr_returns, num_pivots = mc2.ols_monte_carlo(
            trajectory, sampler, tqdm_use, phi, weights, targets, features, x_0, gamma)
        
        # Update the stats.
        total_pivots.append(total_pivots[-1] + num_pivots)
        total_interactions.append(total_interactions[-1] + len(trajectory))
        num_episode.append(num_episode[-1] + 1)
        
        # Store the empirical and predicted returns. For any episode, we may
        # or may not have empirical returns for both actions. When we don't have an
        # estimate, `nan` is returned.
        returns_per_episode_q.append(cur_avr_returns)
        predicted_returns_q.append(np.array(
            [np.inner(x_sa0.flatten(), weights),
                np.inner(x_sa1.flatten(), weights)]))
    
    # The following variant produces plots where we can see
    # the effect of the last update.
    # Do one more evaluation run.
    trajectory, early_term = environments.generate_trajectory(
        env, policy=policy_to_use,
        # env, policy=lambda s: policy(state=s, weights=weights),
        termination_prob=termination_prob, max_env_steps=max_env_steps)
    returns.append(sum(ts[2] for ts in trajectory))

    return {
        "total_return": returns,
        "total_pivots": total_pivots,
        "total_interactions": total_interactions,
        "num_episode": num_episode,
        "returns_per_episode_q": returns_per_episode_q,
        "predicted_returns_q": predicted_returns_q,
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

    date_postfix = datetime.now().strftime("%Y%m%d-%H%M%S")

    results = {}
    for name, sampler in tqdm(samplers_tried.items()):
        print(name, sampler)
        #results[name] = []
        results[name] = Parallel(n_jobs = num_runs)(
            delayed(run_experiment)(
                seed+run, env, phi, sampler, epsilon, budget, budget_type,
                termination_prob, max_env_steps, gamma=gamma, file_postfix=name + "_" + date_postfix,
                tqdm=None, print_trajectory=False, save_threshold=save_limit,
                weights_to_evaluate=weights_to_evaluate)
                for run in range(num_runs)
            )

    print()
    print("DONE!")

    print("Saving results...")
    filename = f"tradeoff_results_{date_postfix}.pkl"
    with open(filename, "wb") as f:
        pickle.dump({"results": results, "config": orig_config}, f)
    print("Saved results.")


