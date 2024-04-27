
import numpy as np
import random
from adaptive_time.environments import cartpole2

def reset_randomness(seed, env):
    """Resets all randomness, except for the environment's seed."""
    random.seed(seed)
    np.random.seed(seed)
    # env.seed(seed)
    if env:
        env.action_space.seed(seed)    


def generate_trajectory(
        env, start_state=None, seed=None, policy=None,
        termination_prob=0.0, max_steps=None):
    """Generates a single trajectory from the environment using the given policy.

    NOTE: the policy gets the current timestep as well!

    Note that the trajectory is generated until the environment terminates, and
    truncations are not handled!
    
    Args:
        env: The environment to generate the trajectory from.
        start_state: The state to start the trajectory from.
        seed: The seed to use for the environment. Defaults to None.
        policy: A function that takes in a state and returns an action. Defaults
            to a random policy if not provided.
        termination_prob: The probability of terminating the trajectory at each
            step. Defaults to 0.0.
        max_steps: The maximum number of steps to take before terminating the
            interaction. Defaults to None.
    
    Returns:
        (traj, is_early_terminated), where
        * traj is list of transitions, where each transition is a tuple of
            `(state, action, reward, next_state)`.
        * is_early_terminated is a boolean indicating if the trajectory was
            terminated early due to `max_steps`.
    """
    if start_state is not None:
        try:
            observation, _ = env.reset(seed=seed, start_state=start_state)
        except TypeError:
            print("start_state is specified, but the environment does not support it!")
            raise
    else:
        observation, _ = env.reset(seed=seed)
    trajectory = []
    terminated = False
    steps = 0
    if policy is None:
        policy = lambda x: env.action_space.sample()
    while not terminated:
        action = policy(observation, steps)
        steps += 1
        observation_, reward, terminated, truncated, info = env.step(action)
        trajectory.append([observation, action, reward, observation_])
        observation = observation_
        if random.random() < termination_prob:
            terminated = True

        if steps % 25_000 == 0:
            print('Did 25_000 steps!', steps)
        
        if max_steps is not None and steps > max_steps:
            return trajectory, True
            #print('Max steps reached!', steps)

    return trajectory, False


def simulate_learning(
        seed, samplers_tried, update_budget, num_runs,
        start_state_weights, approx_integrals, num_pivots, tqdm):
    """Simulates learning in an environment."""
    reset_randomness(seed, env=None)

    data = {}
    num_trajs = len(start_state_weights)

    if tqdm is None:
        tqdm = lambda x: x

    update_in_batches_of = 100_000

    for sampler_name, sampler in tqdm(samplers_tried.items()):
        print("sampler_name:", sampler_name)
        data[sampler_name] = []

        for run_idx in tqdm(range(num_runs)):
            # Main idea: update the value estimate with new samples
            # until we run out of budget.
            used_updates = 0
            # value_estimate = 0
            # num_traj_samples = 0
            cur_data = {
                "values_of_trajs": np.array([]),  # Instantaneous values.
                "pivots_of_trajs": np.array([]),  # Pivots used for the estimate.
                # "running_v_estimate": [],  # Running value estimates.
                # "total_pivots": [],  # Total pivots used for the estimate.
            }

            while used_updates < update_budget:
                # Do updates with np for a batch of 10_000 samples so
                # we can calculate this fast.

                sampled_start_states = np.random.choice(
                    num_trajs, size=(update_in_batches_of,), p=start_state_weights)

                cur_values = approx_integrals[sampler_name][sampled_start_states]
                cur_pivots = num_pivots[sampler_name][sampled_start_states]
                cur_data["values_of_trajs"] = np.concatenate(
                    (cur_data["values_of_trajs"], cur_values))
                cur_data["pivots_of_trajs"] = np.concatenate(    
                    (cur_data["pivots_of_trajs"], cur_pivots))

                used_updates += np.sum(cur_pivots)
                # num_traj_samples += 1
                # start_state = np.random.choice(num_trajs, p=start_state_weights)
                # val_sample = approx_integrals[sampler_name][start_state]
                # cur_data["values_of_trajs"].append(val_sample)
                
                # value_estimate += (1.0/num_traj_samples) * (val_sample - value_estimate)
                # used_updates += num_pivots[sampler_name][start_state]

                # cur_data["running_v_estimate"].append(value_estimate)
                # cur_data["total_pivots"].append(used_updates)
            
            running_value_est = (
                np.cumsum(cur_data["values_of_trajs"])
                / np.arange(1, len(cur_data["values_of_trajs"]) + 1))
            data_to_store = {
                "running_v_estimate": running_value_est,
                "total_pivots": np.cumsum(cur_data["pivots_of_trajs"]),
            }
            data[sampler_name].append(data_to_store)

    return data


def simulate_learning_orig(
        seed, samplers_tried, update_budget, num_runs,
        start_state_weights, approx_integrals, num_pivots, tqdm):
    """Simulates learning in an environment."""
    reset_randomness(seed, env=None)

    num_trajs = len(start_state_weights)

    if tqdm is None:
        tqdm = lambda x: x

    estimated_values_by_episode = {}
    number_of_pivots_by_episode = {}
    all_values_by_episode = {}

    for sampler_name, sampler in tqdm(samplers_tried.items()):
        print("sampler_name:", sampler_name)
        # Update the value estimate with new samples until we run out of budget.
        used_updates = 0
        value_estimate = 0
        num_samples = 0

        all_values_by_episode[sampler_name] = []
        # empirical_state_distr = np.zeros((num_trajs))

        estimated_values_by_episode[sampler_name] = []
        number_of_pivots_by_episode[sampler_name] = []

        while used_updates < update_budget:
            num_samples += 1
            start_state = np.random.choice(num_trajs, p=start_state_weights)
            # empirical_state_distr[start_state] += 1
            val_sample = approx_integrals[sampler_name][start_state]
            all_values_by_episode[sampler_name].append(val_sample)
            
            value_estimate += (1.0/num_samples) * (val_sample - value_estimate)
            used_updates += num_pivots[sampler_name][start_state]

            estimated_values_by_episode[sampler_name].append(value_estimate)
            number_of_pivots_by_episode[sampler_name].append(used_updates)
        
        # empirical_state_distr /= np.sum(empirical_state_distr)
        # empirical_value = approx_integrals[sampler_name] @ empirical_state_distr


    # CODE TO SAMPLE MANY TRAJECOTRIES TO FIND AN EMPIRICAL DISTRIBUTION 
    # episode_samples = 100_000
    # sampled_start_states = np.random.choice(num_trajs, size=(episode_samples,), p=weights)
    # # We now have samples, we determine the empirical state distribution.
    # empirical_state_distr = np.zeros((num_trajs))
    # values, counts = np.unique(sampled_start_states, return_counts=True)
    # empirical_state_distr[values] = counts
    # empirical_state_distr /= np.sum(empirical_state_distr)

    return (
        estimated_values_by_episode,
        number_of_pivots_by_episode,
        all_values_by_episode
    )