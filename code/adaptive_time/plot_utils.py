"""Contains utilities for plotting results."""

import numpy as np


# Could maybe do a variant of this without specifying y_label.
def interpolate_and_stack(
        dict_of_multiple_runs, x_label: str, y_label: str, right=None):
    """"Interpolates and stacks the x vs y.
    
    `dict_of_multiple_runs` has the following structure:
    
    1. The keys are the names of the different methods.
    2. Each value is a list, containing the data_dict of the different 
        runs of the same method.
    3. The data_dict from each run is in a dictionary, mapping labels to 
        the actual data.

    `x_label` selects one of the labels from the data_dict, whose data is
    monotonically increasing; e.g. number of updates, or number of interactions.
    We ensure that the data is interpolated with respect to this label. See
    an example (test or in a notebook) for the usage.
    """
    max_x_value = max(
        max(run[x_label][-1] for run in runs)
        for runs in dict_of_multiple_runs.values())
    all_x_values = np.arange(0, max_x_value+1)   # x-axis for the interpolation

    interpolated_results = {}  # Each element will be a numpy array
    for name, stats_for_runs in dict_of_multiple_runs.items():
        interpolated_results[name] = np.zeros((len(stats_for_runs), len(all_x_values)))
        for run_idx, run in enumerate(stats_for_runs):
            length = min(len(run[x_label]), len(run[y_label]))
            # print("run_x_label:\n", len(run[x_label]))
            # print("run_y_label:\n", len(run[y_label]))
            # print("run_y_label 2:\n", len(run[y_label][:num_x_data_points]))
            # print("run_x_label:\n", run[x_label])
            # print("run_y_label:\n", run[y_label])
            interpolated_results[name][run_idx] = np.interp(
                all_x_values, run[x_label][:length], run[y_label][:length], right=right)

    return all_x_values, interpolated_results


# def pad_along_axis(
#         array: np.ndarray, target_length: int, axis: int = 0, pad_value=0.0
# ) -> np.ndarray:
#     pad_size = target_length - array.shape[axis]
#     if pad_size <= 0:
#         return array
#     npad = [(0, 0)] * array.ndim
#     npad[axis] = (0, pad_size)
#
#     return np.pad(
#         array, pad_width=npad, mode='constant', constant_values=pad_value)


def pad_combine(list_of_vecs, pad_value=np.nan):
    """Pads the vectors in list_of_vecs to the same length and stacks them."""

    max_len = max([len(v) for v in list_of_vecs])
    padded = [
        np.pad(v, (0, max_len - len(v)), mode='constant', constant_values=pad_value)
        # pad_along_axis(v, max_len, axis=0, pad_value=pad_value)
        for v in list_of_vecs
    ]
    return np.stack(padded, axis=1)