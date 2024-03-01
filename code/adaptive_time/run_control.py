from adaptive_time import samplers
from adaptive_time import run_lib


def run_control():
    config = {
        "seed": 13,
        # Save models that get more than this return:
        "save_limit": None,    # 40_000

        # "termination_prob": 1.0/500.0,   # 1.0/500000.0
        "termination_prob": 1.0/10000.0,   # 1.0/500000.0
        "max_env_steps": None,  # Not supported due to truncation issues.
        "epsilon": 0.05,
        "gamma": 0.99999,

        "do_weighing": True,   # Of the updates. Normally True.

        # "budget": 2_000,
        "budget": 10_000,
        "budget_type": run_lib.BudgetType.UPDATES,
        # "budget": 10_000,
        # "budget_type": BudgetType.INTERACTIONS,

        "num_runs": 5,  # Number of runs for each configuration.
        # "tau": 0.002,   # The stepTime of the environment.
        "tau": 0.02,   # The stepTime of the environment.

        # If None, we do control. Otherwise, we evaluate these weights/actions.
        "weights_to_evaluate": None,  # 0 -- will evaluate0 weights, "path", will load the weights from that path
        "policy_to_evaluate": None, # tuple(path to store action sequence (good) , path to stored another action sequence (bad), p = value of probability to select between the policies.)
        # job_lib is true by default and used to parallize

        # "use_joblib": True,  # Whether to use joblib for parallelization.

        # The features cannot be set through a config for now.
        # They are defined in run_lib.py
    }

    # sampler = samplers.AdaptiveQuadratureSampler2(tolerance=0.1)
    # sampler = samplers.AdaptiveQuadratureSampler2(tolerance=0.0)

    samplers_tried = dict(
        # larger tolerance is more coarse approximation for the return. 
        # q0_10=samplers.AdaptiveQuadratureSampler2(tolerance=10),
        # q0_5=samplers.AdaptiveQuadratureSampler2(tolerance=5),
        # q0_1=samplers.AdaptiveQuadratureSampler2(tolerance=1),
        u1=samplers.UniformSampler2(1),
        # u5=samplers.UniformSampler2(5),
        # u10=samplers.UniformSampler2(10),
        # u20=samplers.UniformSampler2(20), # how many steps to skip at each time. 
    )

    run_lib.run_generic(config, samplers_tried)


if __name__ == "__main__":
    run_control()
