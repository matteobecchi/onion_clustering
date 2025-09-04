"""
Code for clustering of multivariate time-series data.
See the documentation for all the details.
"""

# Author: Becchi Matteo <bechmath@gmail.com>

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky
from sklearn.mixture import GaussianMixture

from tropea_clustering._internal.onion_smooth.first_classes import (
    StateMulti,
)
from tropea_clustering._internal.onion_smooth.functions import (
    find_half_height_around_max,
    find_minima_around_max,
    moving_average_2d,
    relabel_states_2d,
)


def gauss_fit_max(
    matrix: NDArray[np.float64],
    tmp_labels: NDArray[np.int64],
    m_limits: NDArray[np.float64],
    bins: int | str,
    number_of_sigmas: float,
) -> StateMulti | None:
    """
    Selection of the optimal region and parameters in order to fit a state.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles, n_frames)
        The time-series data to cluster.

    tmp_labels : ndarray of shape (n_particles, n_frames)
        Temporary labels for each frame. Unclassified points are given
        the label "0".

    m_limits : ndarray
        The min and max of the data points, for each feature.

    bins : int, default="auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigmas : float, default=3.0
        Sets the thresholds for classifing a signal sequence inside a state:
        the sequence is contained in the state if it is entirely contained
        inside number_of_sigmas * state.sigmas times from state.mean.

    Returns
    -------
    state : StateMulti | None
        It is None if the fit failed.
    """
    mask = tmp_labels == 0
    flat_m = matrix[mask]
    if bins == "auto":
        bins = max(int(np.power(matrix.size, 1 / 3) * 2), 10)
    counts, edges = np.histogramdd(flat_m, bins=bins, density=True)

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

    state = StateMulti(popt[0], popt[1], popt[2], number_of_sigmas)

    return state


def find_stable_trj(
    matrix: NDArray[np.float64],
    tmp_labels: NDArray[np.int64],
    state: StateMulti,
    delta_t: int,
    lim: int,
    number_of_sigmas: float,
) -> tuple[np.ndarray, float]:
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
    mask_unclassified = tmp_labels == 0

    m_clean = matrix.copy()
    l_cholesky = cholesky(state.covariance, lower=True)
    l_inv = np.linalg.inv(l_cholesky)
    rescaled = ((m_clean - state.mean) @ l_inv.T) / np.sqrt(matrix.shape[2])
    squared_distances = np.sum(rescaled**2, axis=2)

    mask_dist = squared_distances <= number_of_sigmas**2

    mask = mask_unclassified & mask_dist

    mask_stable = np.zeros_like(tmp_labels, dtype=bool)
    for i, _ in enumerate(matrix):
        row_mask = mask[i]
        padded = np.concatenate(([False], row_mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            if end - start >= delta_t:
                mask_stable[i, start:end] = True

    tmp_labels[mask_stable] = lim + 1
    fraction = np.sum(mask_stable) / mask_stable.size

    return tmp_labels, fraction


def iterative_search(
    matrix: NDArray[np.float64],
    delta_t: int,
    bins: int | str,
    number_of_sigmas: float,
    max_area_overlap: float,
) -> tuple[list[StateMulti], NDArray[np.int64]]:
    """
    Iterative search for stable sequences in the trajectory.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles, n_frames, n_features)
        The time-series data to cluster.

    delta_t : int
        The minimum lifetime required for the clusters.

    bins : int, default="auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigmas : float, default=3.0
        Sets the thresholds for classifing a signal sequence inside a state:
        the sequence is contained in the state if it is entirely contained
        inside number_of_sigmas * state.sigmas times from state.mean.

    max_area_overlap : float, default=0.8
        Thresold to consider two Gaussian states overlapping, and thus merge
        them together.

    Results
    -------
    states_list : List[StateMulti]
        The list of the identified states.

    labels : ndarray of shape (n_particles, n_frames)
        Cluster labels for each frame. Unclassified points are given
        the label "-1".
    """
    tmp_labels = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=int)
    tmp_states_list = []
    states_counter = 0

    min_vals = matrix.min(axis=(0, 1))  # shape: (n_dims,)
    max_vals = matrix.max(axis=(0, 1))  # shape: (n_dims,)
    bounds = np.stack((min_vals, max_vals), axis=1)

    while True:
        state = gauss_fit_max(
            matrix,
            tmp_labels,
            bounds,
            bins,
            number_of_sigmas,
        )
        if state is None:
            break

        tmp_labels, counter = find_stable_trj(
            matrix,
            tmp_labels,
            state,
            delta_t,
            states_counter,
            number_of_sigmas,
        )
        if counter == 0.0:
            break

        state.perc = counter
        tmp_states_list.append(state)
        states_counter += 1

    labels, state_list = relabel_states_2d(
        max_area_overlap,
        tmp_labels,
        tmp_states_list,
    )

    return state_list, labels - 1


def _main(
    matrix: NDArray[np.float64],
    delta_t: int,
    bins: int | str,
    number_of_sigmas: float,
    max_area_overlap: float,
) -> tuple[list[StateMulti], NDArray[np.int64]]:
    """
    Performs onion clustering on the data array 'matrix' at a give delta_t.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles, n_frames, n_features)
        The time-series data to cluster.

    delta_t : int
        The minimum lifetime required for the clusters.

    bins : int, default="auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigmas : float, default=3.0
        Sets the thresholds for classifing a signal sequence inside a state:
        the sequence is contained in the state if it is entirely contained
        inside number_of_sigmas * state.sigmas times from state.mean.

    max_area_overlap : float, default=0.8
        Thresold to consider two Gaussian states overlapping, and thus merge
        them together.

    Returns
    -------
    states_list : List[StateMulti]
        The list of the identified states.

    labels : ndarray of shape (n_particles, n_frames)
        Cluster labels for each frame. Unclassified points are given
        the label "-1".
    """
    tmp_state_list, tmp_labels = iterative_search(
        matrix,
        delta_t,
        bins,
        number_of_sigmas,
        max_area_overlap,
    )

    return tmp_state_list, tmp_labels
