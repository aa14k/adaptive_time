"""Generate all cartpole variant plots.

Follows the example from https://stackoverflow.com/a/48414195.
"""

import sys,os

from tqdm import tqdm
from adaptive_time import utils
import itertools

IPYNB_FILENAME = 'code/adaptive_time/notebooks/value_estimation.ipynb'
CONFIG_FILENAME = '.tmp_config_ipynb'

def main(argv):
    utils.set_directory_in_project()

    # Static settings.
    generate_new_data=False
    save_new_data=False
    num_runs=30
    sample_budget=1000000
    base_args = (
        "run_cartpole.py "
        f"--generate_new_data {generate_new_data} "
        f"--save_new_data {save_new_data} "
        f"--num_runs {num_runs} "
        f"--sample_budget {sample_budget}")

    # Orig code:
    # configs = [' '.join(argv)]

    # configs = [base_args]
    # Dynamic settings.
    configs = []
    for a in itertools.product([False, True],repeat=2):
        config = f"--discrete_reward {a[0]} --terminate_env {a[1]}"
        configs.append(f"{base_args} {config}")    

    for config in tqdm(configs):
        with open(CONFIG_FILENAME,'w') as f:
            f.write(config)
        os.system('jupyter nbconvert --execute {:s} --to html'.format(IPYNB_FILENAME))
    return None

if __name__ == '__main__':
    main(sys.argv)


