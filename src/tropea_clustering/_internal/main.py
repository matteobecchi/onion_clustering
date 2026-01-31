"""Code for Onion clustering of time-series data."""

import copy

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky
from sklearn.mixture import GaussianMixture

from tropea_clustering._internal.classes import OnionData, OnionParams
from tropea_clustering._internal.functions import (
    find_half_height_around_max,
    find_minima_around_max,
    moving_average_2d,
    relabel_states_2d,
)


def gauss_fit_max(
    data: OnionData,
    params: OnionParams,
) -> dict | None:
    """
    Selection of the optimal region and parameters in order to fit a state.

    Parameters
    ----------

    Returns
    -------
    state : dict | None
        It is None if the fit failed.
    """
    mask = data.labels == -1
    flat_m = data.data[mask]
    if params.bins == "auto":
        params.bins = max(int(np.power(data.data.size, 1 / 3) * 2), 10)
    counts, edges = np.histogramdd(flat_m, bins=params.bins, density=True)

    gap = 1
    edges_sides = np.array([e.size for e in edges])
    if np.all(edges_sides > 49):
        gap = int(np.min(edges_sides) * 0.02) * 2
        if gap % 2 == 0:
            gap += 1

    counts = moving_average_2d(counts, gap)

    def find_max_index(data: np.ndarray):
        max_val = data.max()
        max_indices = np.argwhere(data == max_val)
        return max_indices[0]

    max_ind = find_max_index(counts)

    rectangle = find_minima_around_max(counts, max_ind, gap)
    bounds = []
    for dim in range(len(edges)):
        min_idx = rectangle[2 * dim]
        max_idx = rectangle[2 * dim + 1]

        # Use bin edges to get data-space coordinates
        lower = edges[dim][min_idx]
        upper = edges[dim][max_idx + 1]
        bounds.append((lower, upper))
    mask = np.ones(flat_m.shape[0], dtype=bool)
    for dim in range(flat_m.shape[1]):
        lower, upper = bounds[dim]
        mask &= (flat_m[:, dim] >= lower) & (flat_m[:, dim] < upper)

    if flat_m[mask].shape[0] < 2:
        return None  # GMM requires at least 2 samples
    gmm = GaussianMixture(n_components=1, random_state=0).fit(flat_m[mask])
    popt_min = [gmm.means_[0], gmm.covariances_[0], gmm.score(flat_m[mask])]
    flag_min = gmm.converged_

    rectangle = find_half_height_around_max(counts, max_ind, gap)

    bounds = []
    for dim in range(len(edges)):
        min_idx = rectangle[2 * dim]
        max_idx = rectangle[2 * dim + 1]

        # Use bin edges to get data-space coordinates
        lower = edges[dim][min_idx]
        upper = edges[dim][max_idx + 1]
        bounds.append((lower, upper))
    mask = np.ones(flat_m.shape[0], dtype=bool)
    for dim in range(flat_m.shape[1]):
        lower, upper = bounds[dim]
        mask &= (flat_m[:, dim] >= lower) & (flat_m[:, dim] < upper)

    gmm = GaussianMixture(n_components=1, random_state=0).fit(flat_m[mask])
    popt_half = [gmm.means_[0], gmm.covariances_[0], gmm.score(flat_m[mask])]
    flag_half = gmm.converged_

    if flag_min == 1 and flag_half == 0:
        popt = popt_min
    elif flag_min == 0 and flag_half == 1:
        popt = popt_half
    elif flag_min * flag_half == 1:
        if popt_min[2] >= popt_half[2]:
            popt = popt_min
        else:
            popt = popt_half
    else:
        return None

    state = {
        "mean": popt[0],
        "covariance": popt[1],
        "log_likelihood": popt[2],
        "perc": 0.0,
    }

    return state


def find_stable_trj(
    data: OnionData,
    delta_t: int,
    params: OnionParams,
    state: dict,
    states_counter: int,
) -> tuple[OnionData, float]:
    """
    Identification of sequences contained in a certain state.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles, n_frames)
        The time-series data to cluster.

    tmp_labels : ndarray of shape (n_particles, n_frames)
        Temporary labels for each frame. Unclassified points are given
        the label "0".

    state : StateUni
        A Gaussian state.

    delta_t : int
        The minimum lifetime required for the clusters.

    lim : int
        The algorithm iteration.

    number_of_sigmas: float
        Sets the thresholds for classifing a signal sequence inside a state:
        the sequence is contained in the state if it is entirely contained
        inside number_of_sigmas * state.sigmas times from state.mean.

    Returns
    -------
    tmp_labels : ndarray of shape (n_particles, n_frames)
        Updated temporary labels for each frame. Unclassified points are given
        the label "0".

    fraction : float
        Fraction of data points classified in this state.
    """
    mask_unclassified = data.labels == -1

    m_clean = data.data.copy()
    l_cholesky = cholesky(state["covariance"], lower=True)
    l_inv = np.linalg.inv(l_cholesky)
    rescaled = ((m_clean - state["mean"]) @ l_inv.T) / np.sqrt(data.ndims)
    squared_distances = np.sum(rescaled**2, axis=2)

    mask_dist = squared_distances <= params.number_of_sigmas**2

    mask = mask_unclassified & mask_dist

    mask_stable = np.zeros_like(data.labels, dtype=bool)
    for i, _ in enumerate(data.data):
        row_mask = mask[i]
        padded = np.concatenate(([False], row_mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            if end - start >= delta_t:
                mask_stable[i, start:end] = True

    data_copy = copy.deepcopy(data)
    if data_copy.labels is not None:
        data_copy.labels[mask_stable] = states_counter
    fraction = np.sum(mask_stable) / mask_stable.size

    return data_copy, fraction


def perform_onion_clustering(
    data: OnionData,
    delta_t: int,
    params: OnionParams,
) -> tuple[list[dict], NDArray[np.int64]]:
    """The main function, to be written."""
    tmp_state_list = []
    data_copy = copy.deepcopy(data)

    states_counter = 0
    while True:
        state = gauss_fit_max(
            data_copy,
            params,
        )
        if state is None:
            break

        data_copy, fraction = find_stable_trj(
            data_copy,
            delta_t,
            params,
            state,
            states_counter,
        )
        if fraction == 0.0:
            break

        state["perc"] = fraction
        tmp_state_list.append(state)
        states_counter += 1

    print(tmp_state_list)

    labels, state_list = relabel_states_2d(
        params.max_area_overlap,
        data_copy.labels,
        tmp_state_list,
    )

    return state_list, labels
