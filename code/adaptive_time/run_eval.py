from adaptive_time import samplers
from adaptive_time import run_lib


def run_control():


    config = {
        "seed": 13,
        # Save models that get more than this return:
        "save_limit": None,    # 40_000

        # We have to ensure that the policy we evaluate fails eventually!
        "termination_prob": 0,
        "max_env_steps": None,  # Not supported due to truncation issues.
        "epsilon": 0.08,
        "gamma": 0.99999,

        "fourier_order": 4,
        "regularizer": 0.1,
        "do_weighing": True,   # Of the updates. Normally True.

        "budget": 100_000,
        # "budget": 1_000,
        # "budget": 100_000,
        # # "budget": 500,
        "budget_type": run_lib.BudgetType.UPDATES,
        # "budget": 10_000,
        # "budget_type": BudgetType.INTERACTIONS,
        # "budget": 2,
        # "budget_type": run_lib.BudgetType.EPISODES,

        # "num_runs": 5,  # Number of runs for each configuration.
        "num_runs": 1,  # Number of runs for each configuration.
        "tau": 0.02,   # The stepTime of the environment.

        # We may evaluate some fixed weights, or a sequence of actions.
        # Option 1: Evaluate a fixed set of weights.
        #    set to None to not do this.
        #    set to a string to load the weights from a file.
        #    set to 0 to evaluate the 0 vector.
        "weights_to_evaluate": 'cartpole_weights_20240227-102913_ret92516.44719752521.npy',
        # Option 2: Evaluate a sequence of actions.
        #    set to None to not do this.
        #    set to a string to load the sequence of actions from a file.
        "policy_to_evaluate": None,

        # "use_joblib": True,  # Whether to use joblib for parallelization.

        # The features cannot be set through a config for now.
        # They are defined in run_lib.py
    }

    # sampler = samplers.AdaptiveQuadratureSampler2(tolerance=0.1)
    # sampler = samplers.AdaptiveQuadratureSampler2(tolerance=0.0)

    samplers_tried = dict(
        u1=samplers.UniformSampler2(1),
        # u5=samplers.UniformSampler2(5),
        # u10=samplers.UniformSampler2(10),
        # u20=samplers.UniformSampler2(20),
        # q0_10=samplers.AdaptiveQuadratureSampler2(tolerance=10),
        # q0_5=samplers.AdaptiveQuadratureSampler2(tolerance=5),
        # q0_1=samplers.AdaptiveQuadratureSampler2(tolerance=1),
    )

    run_lib.run_generic(config, samplers_tried)

if __name__ == "__main__":
    run_control()