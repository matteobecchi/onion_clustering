"""Should contains supporting functions for the 2 codes."""

# Author: Becchi Matteo <bechmath@gmail.com>

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.stats import chi2

from tropea_clustering._internal.onion_smooth.first_classes import (
    StateMulti,
    StateUni,
)


def moving_average_2d(
    data: NDArray[np.float64],
    side: int,
) -> NDArray[np.float64]:
    """
    2D moving average on an np.ndarray.

    Parameters
    ----------
    data : np.ndarray of shape (a, b)
        The 2D input array to be smoothed.

    side : int
        The side length of the square moving average window (must be
        an odd number).

    Returns
    -------
    np.ndarray
        The smoothed array.
    """
    if side % 2 == 0:
        raise ValueError("L must be an odd number.")

    half_width = (side - 1) // 2
    result = np.zeros_like(data, dtype=float)
    for index in np.ndindex(*data.shape):
        slices = tuple(
            slice(
                max(0, i - half_width),
                min(data.shape[dim], i + half_width + 1),
            )
            for dim, i in enumerate(index)
        )
        subarray = data[slices]
        if subarray.size > 0:
            result[index] = subarray.mean()

    return result


def gaussian(
    x_points: NDArray[np.float64],
    x_mean: float,
    sigma: float,
    area: float,
) -> NDArray[np.float64]:
    """Compute the Gaussian function values at given points 'x_points'.

    Parameters
    ----------
    x_points : np.ndarray
        Array of input values.

    x_mean : float
        Mean value of the Gaussian.

    sigma : float
        Rescaled standard deviation of the Gaussian.

    area : float
        Area under the Gaussian.

    Returns
    -------
    np.ndarray
        Gaussian function values computed at the input points.
    """
    return (
        np.exp(-0.5 * (((x_points - x_mean) / sigma) ** 2))
        * area
        / (np.sqrt(np.pi * 2) * sigma)
    )


def find_minima_around_max(
    data: NDArray[np.float64],
    max_ind: tuple[int, ...],
    gap: int,
) -> list[int]:
    """
    Minima surrounding the maximum value in the given data array.

    Parameters
    ----------
    data : np.ndarray
        Input data array.

    max_ind : tuple
        Indices of the maximum value in the data.

    gap : int
        Gap value to determine the search range around the maximum.

    Returns
    -------
    minima : List[int]
        List of indices representing the minima surrounding the maximum
        in each dimension.
    """
    minima: list[int] = []

    for dim in range(data.ndim):
        min_id0 = max(max_ind[dim] - gap, 0)
        min_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

        tmp_max1: list[int] = list(max_ind)
        tmp_max2: list[int] = list(max_ind)

        tmp_max1[dim] = min_id0
        tmp_max2[dim] = min_id0 - 1
        while min_id0 > 0 and data[tuple(tmp_max1)] > data[tuple(tmp_max2)]:
            tmp_max1[dim] -= 1
            tmp_max2[dim] -= 1
            min_id0 -= 1

        tmp_max1 = list(max_ind)
        tmp_max2 = list(max_ind)

        tmp_max1[dim] = min_id1
        tmp_max2[dim] = min_id1 + 1
        while (
            min_id1 < data.shape[dim] - 1
            and data[tuple(tmp_max1)] > data[tuple(tmp_max2)]
        ):
            tmp_max1[dim] += 1
            tmp_max2[dim] += 1
            min_id1 += 1

        minima.extend([min_id0, min_id1])

    return minima


def find_half_height_around_max(
    data: NDArray[np.float64],
    max_ind: tuple[int, ...],
    gap: int,
) -> list[int]:
    """
    Half-heigth points surrounding the maximum value in the given data array.

    Parameters
    ----------
    data : np.ndarray
        Input data array.

    max_ind : tuple
        Indices of the maximum value in the data.

    gap : int
        Gap value to determine the search range around the maximum.

    Returns
    -------
    minima : list[int]
        List of indices representing the minima surrounding the maximum
        in each dimension.
    """
    max_val = data.max()
    minima: list[int] = []

    for dim in range(data.ndim):
        half_id0 = max(max_ind[dim] - gap, 0)
        half_id1 = min(max_ind[dim] + gap, data.shape[dim] - 1)

        tmp_max: list[int] = list(max_ind)

        tmp_max[dim] = half_id0
        while half_id0 > 0 and data[tuple(tmp_max)] > max_val / 2:
            tmp_max[dim] -= 1
            half_id0 -= 1

        tmp_max = list(max_ind)

        tmp_max[dim] = half_id1
        while (
            half_id1 < data.shape[dim] - 1
            and data[tuple(tmp_max)] > max_val / 2
        ):
            tmp_max[dim] += 1
            half_id1 += 1

        minima.extend([half_id0, half_id1])

    return minima


def relabel_states(
    all_the_labels: NDArray[np.int64],
    states_list: list[StateUni],
) -> tuple[NDArray[np.int64], list[StateUni]]:
    """
    Relabel states and update the state list.

    Parameters
    ----------
    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels assigned to each window in the trajectory.

    states_list : List[StateUni]
        List of StateUni objects representing different states.

    Returns
    -------
    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels assigned to each window in the trajectory.

    relevant_states : List[StateUni]
        Updated list of StateUni objects representing different states.
    """
    relevant_states = [state for state in states_list if state.perc != 0.0]

    relevant_states.sort(key=lambda x: x.mean)

    state_mapping = {
        state_index: index + 1
        for index, state_index in enumerate(
            [states_list.index(state) for state in relevant_states]
        )
    }
    relabel_map = np.zeros(len(states_list) + 1, dtype=int)
    for key, value in state_mapping.items():
        relabel_map[key + 1] = value

    mask = all_the_labels != 0
    all_the_labels[mask] = relabel_map[all_the_labels[mask]]

    return all_the_labels, relevant_states


def find_intersection(st_0: StateUni, st_1: StateUni) -> tuple[float, int]:
    """
    Finds the intersection between two Gaussians.

    Parameters
    ----------
    st_0, st_1 : StateUni
        The two states we are computing the intersection between.

    Returns
    -------
    th_val : float
        The value of the intersection.

    th_type : int
        The type of the intersection: type 1 if the intersection between the
        two Gaussians exists, type 2 if it does not exist and the weighted
        average between the means is returned.
    """
    coeff_a = st_1.sigma**2 - st_0.sigma**2
    coeff_b = -2 * (st_0.mean * st_1.sigma**2 - st_1.mean * st_0.sigma**2)
    tmp_c = np.log(st_0.area * st_1.sigma / st_1.area / st_0.sigma)
    coeff_c = (
        (st_0.mean * st_1.sigma) ** 2
        - (st_1.mean * st_0.sigma) ** 2
        - ((st_0.sigma * st_1.sigma) ** 2) * tmp_c
    )
    delta = coeff_b**2 - 4 * coeff_a * coeff_c
    if coeff_a == 0.0:
        only_th = (st_0.mean + st_1.mean) / 2 - st_0.sigma**2 / 2 / (
            st_1.mean - st_0.mean
        ) * np.log(st_0.area / st_1.area)
        return only_th, 1
    if delta >= 0:
        th_plus = (-coeff_b + np.sqrt(delta)) / (2 * coeff_a)
        th_minus = (-coeff_b - np.sqrt(delta)) / (2 * coeff_a)
        intercept_plus = gaussian(th_plus, st_0.mean, st_0.sigma, st_0.area)
        intercept_minus = gaussian(th_minus, st_0.mean, st_0.sigma, st_0.area)
        if intercept_plus >= intercept_minus:
            return th_plus, 1
        return th_minus, 1
    th_aver = (st_0.mean / st_0.sigma + st_1.mean / st_1.sigma) / (
        1 / st_0.sigma + 1 / st_1.sigma
    )
    return th_aver, 2


def shared_area_between_gaussians(
    area1: float,
    mean1: float,
    sigma1: float,
    area2: float,
    mean2: float,
    sigma2: float,
) -> tuple[float, float]:
    """
    Computes the shared area between two Gaussians.

    Parameters
    ----------
    area1, mean1, sigma1 : float
        The parameters of Gaussian 1.

    area2, mean2, sigma2 : float
        The parameters of Gaussian 2.

    Returns
    -------
    shared_fraction_1 : float
        The fraction of the area of the first Gaussian in common with the
        second Gaussian.

    shared_fraction_2 : float
        The fraction of the area of the second Gaussian in common with the
        first Gaussian.
    """

    def gauss_1(x):
        gauss = gaussian(x, mean1, sigma1, area1)
        return gauss

    def gauss_2(x):
        gauss = gaussian(x, mean2, sigma2, area2)
        return gauss

    def min_of_gaussians(x):
        min_values = np.minimum(
            gaussian(x, mean1, sigma1, area1),
            gaussian(x, mean2, sigma2, area2),
        )
        return min_values

    area_gaussian_1, _ = quad(
        gauss_1, int(mean1 - 3 * sigma1) - 1, int(mean1 + 3 * sigma1) + 1
    )
    area_gaussian_2, _ = quad(
        gauss_2, int(mean2 - 3 * sigma2) - 1, int(mean2 + 3 * sigma2) + 1
    )

    x_min = int(np.min([mean1 - 3 * sigma1, mean2 - 3 * sigma2])) - 1
    x_max = int(np.max([mean1 + 3 * sigma1, mean2 + 3 * sigma2])) + 1
    shared_area, _ = quad(min_of_gaussians, x_min, x_max)

    if area_gaussian_1 > 0.0:
        shared_fraction_1 = shared_area / area_gaussian_1
    else:
        shared_fraction_1 = 0

    if area_gaussian_2 > 0.0:
        shared_fraction_2 = shared_area / area_gaussian_2
    else:
        shared_fraction_2 = 0

    return shared_fraction_1, shared_fraction_2


def final_state_settings(
    list_of_states: list[StateUni],
    m_range: NDArray[np.float64],
) -> list[StateUni]:
    """
    Calculate the final threshold values based on the intercept between
    neighboring states.

    Parameters
    ----------
    list_of_states : list[StateUni]
        The list of final states.

    m_range : np.ndarray of shape (2,)
        Range of values in the data matrix.

    Returns
    -------
    list_of_states : list[StateUni]
        Now with the correct thresholds asssigned to each state.
    """
    if len(list_of_states) == 0:
        return list_of_states

    list_of_states[0].th_inf[0] = m_range[0]
    list_of_states[0].th_inf[1] = 0

    for i in range(len(list_of_states) - 1):
        th_val, th_type = find_intersection(
            list_of_states[i], list_of_states[i + 1]
        )
        list_of_states[i].th_sup[0] = th_val
        list_of_states[i].th_sup[1] = th_type
        list_of_states[i + 1].th_inf[0] = th_val
        list_of_states[i + 1].th_inf[1] = th_type

    list_of_states[-1].th_sup[0] = m_range[1]
    list_of_states[-1].th_sup[1] = 0

    return list_of_states


def set_final_states(
    list_of_states: list[StateUni],
    all_the_labels: np.ndarray,
    area_max_overlap: float,
) -> tuple[list[StateUni], NDArray[np.int64]]:
    """
    Assigns final states and relabels labels based on specific criteria.

    Parameters
    ----------
    list_of_states : List[StateUni]
        List of StateUni objects representing potential states.

    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels for each data point.

    m_range : ndarray of shape (2,)
        Minimum and maximum of values in the data.

    Returns
    -------
    updated_states : List[StateUni]
        The final list of states.

    all_the _labels : ndarray of shape (n_particles, n_windows)
        The final data labels.
    """
    # Find all the possible merges: j could be merged into i --> [j, i]
    proposed_merge = []
    for i, st_0 in enumerate(list_of_states):
        for j, st_1 in enumerate(list_of_states):
            if j > i:
                # Condition 1: area overlap
                shared_area_1, shared_area_2 = shared_area_between_gaussians(
                    st_1.area,
                    st_1.mean,
                    st_1.sigma,
                    st_0.area,
                    st_0.mean,
                    st_0.sigma,
                )
                thresh = area_max_overlap
                if shared_area_1 > thresh >= shared_area_2:
                    proposed_merge.append([j, i])
                elif shared_area_2 > thresh >= shared_area_1:
                    proposed_merge.append([i, j])
                elif shared_area_1 > thresh and shared_area_2 > thresh:
                    proposed_merge.append(
                        [j, i] if shared_area_1 > shared_area_2 else [i, j]
                    )
                # Condition 2: mean proximity
                elif (
                    st_0.peak > st_1.peak
                    and np.abs(st_0.mean - st_1.mean) < st_0.sigma
                    and st_1.sigma < 2 * st_0.sigma
                ):
                    proposed_merge.append([j, i])
                elif (
                    st_1.peak > st_0.peak
                    and np.abs(st_0.mean - st_1.mean) < st_1.sigma
                    and st_0.sigma < 2 * st_1.sigma
                ):
                    proposed_merge.append([i, j])

    # Find the best merges (merge into the closest candidate)
    best_merge = []
    states_to_be_merged = np.unique([pair[0] for pair in proposed_merge])
    for j in states_to_be_merged:
        candidate_merge = []
        for pair in proposed_merge:
            if pair[0] == j:
                candidate_merge.append(pair)
        if len(candidate_merge) == 1:
            best_merge.append(candidate_merge[0])
        else:
            importance = [
                list_of_states[pair[1]].perc for pair in candidate_merge
            ]
            best_merge.append(candidate_merge[np.argmax(importance)])

    # Settle merging chains
    # if [i, j], all the [k, i] become [k, j]
    for pair in best_merge:
        for j, elem in enumerate(best_merge):
            if elem[1] == pair[0] and elem[0] != pair[1]:
                best_merge[j][1] = pair[1]

    # Relabel the labels in all_the_labels
    relabel_dic = {}
    for pair in best_merge:
        relabel_dic[pair[0]] = pair[1]
    _ = np.any(np.unique(all_the_labels) == -1)

    relabel_map = np.zeros(max(np.unique(all_the_labels) + 1), dtype=int)
    for i, _ in enumerate(relabel_map):
        relabel_map[i] = i
    for key, value in relabel_dic.items():
        relabel_map[key] = value

    mask = all_the_labels >= 0
    new_labels = all_the_labels.copy()
    new_labels[mask] = relabel_map[all_the_labels[mask]]

    valid_labels = np.unique(new_labels[new_labels != -1])
    new_ids = np.arange(len(valid_labels))
    remap_dict = dict(zip(valid_labels, new_ids))

    def relabel(x):
        return remap_dict.get(x, -1)  # keep -1 as is

    vectorized_relabel = np.vectorize(relabel)
    all_the_labels = vectorized_relabel(new_labels)

    # Remove merged states from the state list
    states_to_remove = set(s0 for s0, s1 in best_merge)
    updated_states = [
        state
        for i, state in enumerate(list_of_states)
        if i not in states_to_remove
    ]

    for i, state in enumerate(updated_states):
        state.perc = np.sum(all_the_labels == i) / all_the_labels.size

    return updated_states, all_the_labels


def gaussian_overlap_fraction(
    mean1, cov1, mean2, cov2, prob=0.95, n_samples=5000
):
    """
    Estimate fraction of Gaussian 1's contour overlapped by Gaussian 2's contour.
    """
    dim = len(mean1)
    chi2_val = chi2.ppf(prob, df=dim)

    # Sample from first Gaussian
    points = np.random.multivariate_normal(mean1, cov1, size=n_samples)

    # Mahalanobis distance to mean2
    diff = points - mean2
    inv_cov2 = np.linalg.inv(cov2)
    m_dist_sq_2 = np.einsum("ij,jk,ik->i", diff, inv_cov2, diff)

    # Fraction inside contour of Gaussian 2
    inside_2 = np.sum(m_dist_sq_2 <= chi2_val) / n_samples
    return inside_2


def relabel_states_2d(
    all_the_labels: np.ndarray,
    states_list: list[StateMulti],
) -> tuple[NDArray[np.int64], list[StateMulti]]:
    """
    Reorders labels and merges strongly overlapping states in a
    multidimensional space.

    Parameters
    ----------
    all_the_labels : ndarray of shape (n_particles, n_windows)
        Array containing labels for each data point.

    states_list : List[StateMulti]
        List of StateUni objects representing potential states.

    Returns
    -------
    all_the_labels : ndarray of shape (n_particles, n_windows)
        The final data labels.

    updated_states: List[StateMulti]
        The final list of states.
    """
    sorted_states = [state for state in states_list if state.perc != 0.0]

    # Create a dictionary to map old state labels to new ones
    state_mapping = {
        index: i + 1
        for i, index in enumerate(
            np.argsort([-state.perc for state in sorted_states])
        )
    }

    sorted_states.sort(key=lambda x: x.perc, reverse=True)

    # Relabel the data in all_the_labels according to the new states_list
    mask = all_the_labels != 0  # Create a mask for non-zero elements
    all_the_labels[mask] = np.vectorize(state_mapping.get, otypes=[int])(
        all_the_labels[mask] - 1, 0
    )

    # Find all the possible merges
    proposed_merge = []
    for i, st_0 in enumerate(sorted_states):
        for j, st_1 in enumerate(sorted_states):
            if j > i:
                overlap_frac = gaussian_overlap_fraction(
                    st_0.mean,
                    st_0.covariance,
                    st_1.mean,
                    st_1.covariance,
                    prob=0.95,
                    n_samples=5000,
                )
                if overlap_frac >= 0.5:  # e.g., at least 50% overlap
                    proposed_merge.append([j, i])

    # Find the best merges (merge into the closest candidate)
    best_merge = []
    states_to_be_merged = np.unique([pair[0] for pair in proposed_merge])
    for j in states_to_be_merged:
        candidate_merge = []
        for pair in proposed_merge:
            if pair[0] == j:
                candidate_merge.append(pair)
        if len(candidate_merge) == 1:
            best_merge.append(candidate_merge[0])
        else:
            importance = [
                sorted_states[pair[1]].perc for pair in candidate_merge
            ]
            best_merge.append(candidate_merge[np.argmax(importance)])

    # Settle merging chains
    for pair in best_merge:
        for j, elem in enumerate(best_merge):
            if elem[1] == pair[0] and elem[0] != pair[1]:
                best_merge[j][1] = pair[1]

    # Relabel the labels in all_the_labels
    relabel_dic = {}
    for pair in best_merge:
        relabel_dic[pair[0]] = pair[1]
    relabel_map = np.zeros(max(np.unique(all_the_labels) + 1), dtype=int)
    for i, _ in enumerate(relabel_map):
        relabel_map[i] = i
    for key, value in relabel_dic.items():
        relabel_map[key + 1] = value + 1

    # Remove the gaps in the labeling
    tmp_labels = np.unique(relabel_map)
    map2 = np.zeros(max(tmp_labels + 1), dtype=int)
    for i, elem in enumerate(tmp_labels):
        map2[elem] = i
    for i, elem in enumerate(relabel_map):
        relabel_map[i] = map2[elem]

    all_the_labels = relabel_map[all_the_labels.flatten()].reshape(
        all_the_labels.shape
    )

    # Remove merged states from the state list
    states_to_remove = set(s0 for s0, s1 in best_merge)
    updated_states = [
        state
        for i, state in enumerate(sorted_states)
        if i not in states_to_remove
    ]

    # Compute the fraction of data points in each state
    for st_id, state in enumerate(updated_states):
        num_of_points = np.sum(all_the_labels == st_id + 1)
        state.perc = num_of_points / all_the_labels.size

    return all_the_labels, updated_states
