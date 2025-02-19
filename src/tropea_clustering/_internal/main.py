"""Code for clustering of univariate time-series data.

* Author: Becchi Matteo <bechmath@gmail.com>
* Date: February 18, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from typing import List

import numpy as np
from sklearn.mixture import GaussianMixture

from tropea_clustering._internal.first_classes import StateUni


def _find_optimal_gmm_dynamic(
    windows: NDArray[np.float64],
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


def _assign_env0_cluster(responsibilities, threshold):
    """Points with responsibility lower than threshold in the ENV0 cluster."""
    n_points = responsibilities.shape[0]
    env0_labels = np.full(n_points, 0)
    for i in range(n_points):
        max_responsibility = np.max(responsibilities[i])
        if max_responsibility < threshold:
            env0_labels[i] = 1
    return env0_labels


def _sort_states(
    state_list: List[StateUni],
    labels: NDArray[np.int64],
) -> tuple[List[StateUni], NDArray[np.int64]]:
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
    windows: NDArray[np.float64],
    resp_th: float,
) -> tuple[List[StateUni], NDArray[np.int64]]:
    """Performs Onion CLustering with GMM implementation.

    Parameters
    ----------
    time_series : np.ndarray of shape (n_particles, n_frames)
        The signals to be clustered.

    resp_th : float
        Sets the threshold for assigning the windows to a state.

    Returns
    -------
    tuple[List[StateUni], np.ndarray[int, Any]]
        - List of gaussian states.
        - Array with cluster labels
    """

    optimal_k, _ = _find_optimal_gmm_dynamic(windows)
    gmm = GaussianMixture(
        n_components=optimal_k, covariance_type="full", random_state=42
    )
    gmm.fit(windows)

    means = np.mean(gmm.means_, axis=1)
    sigmas = np.array(
        [np.sqrt(np.trace(cov) / cov.shape[0]) for cov in gmm.covariances_]
    )
    g_param = np.transpose([means, sigmas, gmm.weights_])

    responsibilities = gmm.predict_proba(windows)
    labels = gmm.predict(windows)
    env0_labels = _assign_env0_cluster(responsibilities, threshold=resp_th)
    labels[env0_labels == 1] = -1

    # labels = assign_windows(means, sigmas, windows, number_of_sigma)

    state_list = [StateUni(*elem) for elem in g_param]

    state_list, labels = _sort_states(state_list, labels)

    for label in np.unique(labels):
        if label > -1:
            state_list[label].perc = np.sum(labels == label) / labels.size

    return state_list, labels
