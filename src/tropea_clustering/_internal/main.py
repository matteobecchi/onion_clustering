"""Code for clustering of univariate time-series data.

Author: Becchi Matteo <bechmath@gmail.com>
Date: February 18, 2025
"""

from typing import Any, List

import numpy as np
from sklearn.mixture import GaussianMixture

from tropea_clustering._internal.first_classes import StateUni


def find_optimal_gmm_dynamic(
    windows: np.ndarray,
    threshold=0.01,
) -> tuple[int, list[float]]:
    """Dynamic optimization of the number of components based on BIC.

    Parameter
    ---------
    windows : np.ndarray
        List of signal windows.

    threshold : float = 0.01
        Minimum relative increase in BIC for considering one more cluster valid.

    Return
    ------
    tuple[int, list[float]]:
        - The optimal number of clusters
        - The list of BIC for each clusters number
    """
    bic_scores = []
    k = 1
    prev_bic = float("inf")
    while True:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
        )
        gmm.fit(windows)
        current_bic = gmm.bic(windows)
        bic_scores.append(current_bic)

        if k > 1:
            if (current_bic - prev_bic) / prev_bic < threshold:
                break

        prev_bic = current_bic
        k += 1

    optimal_k = k - 1
    return optimal_k, bic_scores


def assign_windows(
    means: np.ndarray,
    sigmas: np.ndarray,
    windows: np.ndarray,
    number_of_sigma: float,
) -> np.ndarray:
    """Assign each signal window to its environment.

    Parameters
    ----------
    means : np.ndarray of shape (n_clusters,)
        Mean value of each cluster.

    sigmas : np.ndarray of shape (n_clusters,)
        Standard deviation of each cluster.

     windows : np.ndarray
        List of signal windows.

    number_of_sigma : float
        Number of standard deviations for the range check.

    Returns
    -------
    cluster_labels : np.ndarray of shape (n_windows,)
        Label for each window. Unclassified windows are labelled -1.
    """
    means = means[:, None, None]  # Shape: (n_clusters, 1, 1)
    sigmas = sigmas[:, None, None]  # Shape: (n_clusters, 1, 1)

    lower_bounds = means - number_of_sigma * sigmas
    upper_bounds = means + number_of_sigma * sigmas

    windows = windows[None, :, :]  # Shape: (1, n_windows, window_length)
    within_bounds = (windows >= lower_bounds) & (windows <= upper_bounds)
    fully_contained = np.all(within_bounds, axis=2)
    cluster_labels = np.argmax(fully_contained, axis=0)
    cluster_labels[~np.any(fully_contained, axis=0)] = -1

    return np.array(cluster_labels)


def sort_states(
    state_list: List[StateUni],
    labels: np.ndarray,
) -> tuple[List[StateUni], np.ndarray[int, Any]]:
    """Sort states according to their mean."""
    sorted_list = sorted(state_list, key=lambda state: state.mean)
    sorted_indices = sorted(
        range(len(state_list)), key=lambda i: state_list[i].mean
    )
    index_mapping = {old: new for new, old in enumerate(sorted_indices)}

    sorted_labels = np.array(
        [index_mapping[label] if label != -1 else -1 for label in labels]
    )

    return sorted_list, sorted_labels


def _onion_inner(
    windows: np.ndarray,
    number_of_sigma: float,
) -> tuple[List[StateUni], np.ndarray[int, Any]]:
    """Performs Onion CLustering with GMM implementation.

    Parameters
    ----------
    time_series : np.ndarray of shape (n_particles, n_frames)
        The signals to be clustered.

    number_of_sigma : float
        Sets the threshold for assigning the windows to a state.

    Returns
    -------
    tuple[List[StateUni], np.ndarray[int, Any]]
        - List of gaussian states.
        - Array with cluster labels
    """

    optimal_k, _ = find_optimal_gmm_dynamic(windows)
    gmm = GaussianMixture(
        n_components=optimal_k, covariance_type="full", random_state=42
    )
    gmm.fit(windows)

    means = np.mean(gmm.means_, axis=1)
    sigmas = np.array(
        [np.sqrt(np.trace(cov) / cov.shape[0]) for cov in gmm.covariances_]
    )
    (means, sigmas) = zip(*sorted(zip(means, sigmas)))
    means = np.asarray(means, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    g_param = np.array([means, sigmas, gmm.weights_]).T

    labels = assign_windows(means, sigmas, windows, number_of_sigma)

    state_list = [StateUni(*elem) for elem in g_param]

    # state_list, labels = sort_states(state_list, labels)

    for label in np.unique(labels):
        if label > -1:
            state_list[label].perc = np.sum(labels == label) / labels.size

    return state_list, labels
