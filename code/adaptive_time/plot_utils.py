"""Contains utilities for plotting results."""

from typing import Dict, NamedTuple

import numpy as np
import matplotlib.pyplot as plt



class ProcData(NamedTuple):
    # A sequence of x values.
    xs: np.ndarray  # shape: (num_x_values,)
    # The mean of the y values for each x value.
    means: Dict[str, np.ndarray]  # shape: (num_x_values,)
    # The std err of the y values for each x value.
    stderrs: Dict[str, np.ndarray]  # shape: (num_x_values,)
    # All the y values (for each run) for each x value.
    all_runs_data: Dict[str, np.ndarray]  # shape: (num_runs, num_x_values)
    # The number of runs.
    num_runs: int
    # The number of methods; the number of keys in the dictionaries.
    num_methods: int



# Could maybe do a variant of this without specifying y_label?
def process_across_runs(
        dict_of_multiple_runs, x_label: str, y_label: str, right=None
) -> ProcData:
    """"Interpolates and stacks the x vs y, returning means and stderrs, too.
    
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

    num_runs = len(next(iter(dict_of_multiple_runs.values())))
    interpolated_results = {}  # Each element will be a numpy array
    for name, stats_for_runs in dict_of_multiple_runs.items():
        interpolated_results[name] = np.zeros((len(stats_for_runs), len(all_x_values)))
        for run_idx, run in enumerate(stats_for_runs):
            length = min(len(run[x_label]), len(run[y_label]))
            interpolated_results[name][run_idx] = np.interp(
                all_x_values, run[x_label][:length], run[y_label][:length],
                right=right)
        assert num_runs == len(stats_for_runs)

    all_y_means = {}
    all_y_stderrs = {}
    for name, res in interpolated_results.items():
        all_y_means[name] = np.nanmean(res, axis=0)
        all_y_stderrs[name] =  np.nanstd(res, axis=0) / np.sqrt(num_runs)

    return ProcData(
        xs=all_x_values,
        means=all_y_means,
        stderrs=all_y_stderrs,
        all_runs_data=interpolated_results,
        num_runs=num_runs,
        num_methods=len(interpolated_results))


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


def plot_stuff(tuples_of_x_y_labels_kwargs, title, show, ax=None):

    ax = plt.gca() if ax is None else ax
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for x, y, label, kwargs in tuples_of_x_y_labels_kwargs:
        plt.plot(x, y, label=label, **kwargs)

    plt.legend()

    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    return ax


def default_plot_per_run_from_dict(
        results: Dict, x_label, y_label, title=None, runs=None, show=True, ax=None):
    """Plot y vs x, potentially for a subset of the runs."""
    if runs is None:
        num_rums = len(next(iter(results.values())))
        runs = list(range(num_rums))
    
    if title is None:
        title = f"{y_label} vs {x_label}"

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if len(results) > len(colors):
        # NOTE this could just be a warning; feel free to change.
        raise ValueError("Too many results to plot.")

    tuples_of_x_y_labels_kwargs = []
    for i, (name, all_seeds_data) in enumerate(results.items()):
        for run_idx in runs:
            seed_data = all_seeds_data[run_idx]
            length = min(len(seed_data[x_label]), len(seed_data[y_label]))
            tuples_of_x_y_labels_kwargs.append((
                seed_data[x_label][:length],
                seed_data[y_label][:length],
                name if run_idx==0 else None,
                # {"color": colors[i], "marker": ".",
                #  "linestyle": "None", "markersize": 5,
                #  "alpha": 0.8}
                {"color": colors[i], "marker": ".",
                 "linestyle": "-", "markersize": 5,
                 "alpha": 0.8}
            ))

    ax = plot_stuff(tuples_of_x_y_labels_kwargs, title, False, ax=ax)
    ax.set_ylabel(y_label, rotation=90, labelpad=5)

    ax.set_xlabel(x_label)
    if show:
        plt.show()
    return ax


def default_plot_per_run_from_procdata(
        proc_data: ProcData, x_plot_label, y_plot_label,
        title=None, runs=None, show=True, ax=None):
    """Plot y vs x, potentially for a subset of the runs."""
    if runs is None:
        runs = list(range(proc_data.num_rums))
    
    if title is None:
        title = f"{y_plot_label} vs {x_plot_label}"

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if proc_data.num_methods > len(colors):
        # NOTE this could just be a warning; feel free to change.
        raise ValueError("Too many results to plot.")

    tuples_of_x_y_labels_kwargs = []
    for i, (name, mean_returns) in enumerate(proc_data.means.items()):
        tuples_of_x_y_labels_kwargs.append((
            proc_data.xs, mean_returns, name,
            # {"color": colors[i], "marker": ".",
            #  "linestyle": "None", "markersize": 5, "alpha": 0.8}
            {"color": colors[i], "marker": "", "linestyle": "-"}
        ))

    tuples_of_x_y_labels_kwargs = []
    for i, (name, all_seeds_data) in enumerate(proc_data.all_runs_data.items()):
        for run_idx in runs:
            seed_data = all_seeds_data[run_idx]
            tuples_of_x_y_labels_kwargs.append((
                proc_data.xs,
                seed_data,
                name if run_idx==0 else None,
                # {"color": colors[i], "marker": ".",
                #  "linestyle": "None", "markersize": 5,
                #  "alpha": 0.8}
                {"color": colors[i], "marker": ".",
                 "linestyle": "-", "markersize": 5,
                 "alpha": 0.8}
            ))

    ax = plot_stuff(tuples_of_x_y_labels_kwargs, title, False, ax=ax)
    ax.set_ylabel(y_plot_label, rotation=90, labelpad=5)
    ax.set_xlabel(x_plot_label)

    if show:
        plt.show()
    return ax


def default_plot_mean_from_proc_data(
        proc_data: ProcData, x_plot_label, y_plot_label,
        title=None, show=True, ax=None):
    """Plot mean y vs x."""
    if title is None:
        title = f"{y_plot_label} vs {x_plot_label}"

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if proc_data.num_methods > len(colors):
        # NOTE this could just be a warning; feel free to change.
        raise ValueError("Too many results to plot.")

    tuples_of_x_y_labels_kwargs = []
    for i, (name, mean_returns) in enumerate(proc_data.means.items()):
        tuples_of_x_y_labels_kwargs.append((
            proc_data.xs, mean_returns, name,
            # {"color": colors[i], "marker": ".",
            #  "linestyle": "None", "markersize": 5, "alpha": 0.8}
            {"color": colors[i], "marker": "", "linestyle": "-"}
        ))

    ax = plot_stuff(tuples_of_x_y_labels_kwargs, title, False, ax=ax)
    ax.set_ylabel(y_plot_label, rotation=90, labelpad=5)
    ax.set_xlabel(x_plot_label)
    if show:
        plt.show()
    return ax


def plot_with_error_bars(
        x_values, y_values, y_stds, label, color, ax, alpha=0.2, linestyle='-'):
    """Plots the mean with error bars. UNTESTED?!"""
    ax.plot(x_values, y_values, label=label, color=color, linestyle=linestyle)
    ax.fill_between(
        x_values, y_values - y_stds, y_values + y_stds,
        color=color, alpha=alpha)