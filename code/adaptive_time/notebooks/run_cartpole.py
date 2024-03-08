"""Generate all cartpole variant plots.

Follows the example from https://stackoverflow.com/a/48414195.
"""

import sys,os
import copy

from tqdm import tqdm
from adaptive_time import utils
import itertools

IPYNB_FILENAME = 'code/adaptive_time/notebooks/value_estimation.ipynb'
CONFIG_FILENAME = '.tmp_config_ipynb'

def main(argv):
    utils.set_directory_in_project()

    # Static settings.
    base_config = {
        "generate_new_data": False,
        "save_new_data": True,
        "num_runs": 30,
        # "num_runs": 30,
        "sample_budget": 2_000_000,
        # "sample_budget": 1_000,
        "discrete_reward": True,
        "terminate_env": True,
    }

    # Orig code:
    # configs = [' '.join(argv)]

    # Dynamic settings.
    search_configs = []
    # for a in itertools.product([False, True],repeat=2):
    #     config = {"discrete_reward": a[0], "terminate_env": a[1]}
    #     search_configs.append(config)
    search_configs.append({"discrete_reward": True, "terminate_env": True})
    search_configs.append({"discrete_reward": True, "terminate_env": False})
    search_configs.append({"discrete_reward": False, "terminate_env": False})
    # search_configs = [{}]

    if not search_configs:
        search_configs = [{}]

    for config in tqdm(search_configs):
        cc = copy.deepcopy(base_config)
        cc.update(config)
        # Turn it into a flag string string.
        args = 'run_cartpole.py ' + ' '.join([f"--{k} {v}" for k,v in cc.items()])
        print("Run:", args)
        with open(CONFIG_FILENAME,'w') as f:
            f.write(args)
        os.system('jupyter nbconvert --execute {:s} --to html'.format(IPYNB_FILENAME))
    return None

if __name__ == '__main__':
    main(sys.argv)


