"""
Code for clustering of univariate time-series data.
See the documentation for all the details.
"""

# Author: Becchi Matteo <bechmath@gmail.com>

import warnings
from typing import Tuple

import numpy as np
import scipy.signal
from numpy.typing import NDArray
from scipy.optimize import OptimizeWarning
from scipy.stats import gaussian_kde

from tropea_clustering._internal.classes import ClusteringObject1D
from tropea_clustering._internal.first_classes import (
    Parameters,
    StateUni,
)
from tropea_clustering._internal.functions import (
    gaussian,
    relabel_states,
    set_final_states,
)


def perform_gauss_fit(
    param: list[int], data: list[np.ndarray], int_type: str
) -> Tuple[bool, int, np.ndarray, np.ndarray]:
    """
    Gaussian fit on the data histogram.

    Parameters
    ----------

    param : List[int]
        A list of the parameters for the fit:
            initial index,
            final index,
            index of the max,
            amount of data points,
            gap value for histogram smoothing

    data : List[np.ndarray]
        A list of the data for the fit:
            histogram binning,
            histogram counts

    int_type : str
        The type of the fitting interval ('max' or 'half').

    Returns
    -------

    A boolean value for the fit convergence.

    goodness : int
        The fit quality (max is 5).

    popt : ndarray of shape (3,)
        The optimal gaussians fit parameters.

    Notes
    -----

    - Trys to perform the fit with the specified parameters
    - Computes the fit quality by checking if some requirements are satisfied
    - If the fit fails, returns (False, 5, np.empty(3))
    """
    ### Initialize return values ###
    flag = False
    coeff_det_r2 = 0
    popt = np.empty(3)
    perr = np.empty(3)

    id0, id1, max_ind, n_data = param
    bins, counts = data

    selected_bins = bins[id0:id1]
    selected_counts = counts[id0:id1]
    mu0 = bins[max_ind]
    sigma0 = (bins[id0] - bins[id1]) / 6
    area0 = counts[max_ind] * np.sqrt(np.pi) * sigma0
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            popt, pcov, infodict, _, _ = scipy.optimize.curve_fit(
                gaussian,
                selected_bins,
                selected_counts,
                p0=[mu0, sigma0, area0],
                full_output=True,
            )
            if popt[1] < 0:
                popt[1] = -popt[1]
                popt[2] = -popt[2]
            popt[2] *= n_data
            perr = np.array([np.sqrt(pcov[i][i]) for i in range(popt.size)])
            perr[2] *= n_data
            ss_res = np.sum(infodict["fvec"] ** 2)
            ss_tot = np.sum((selected_counts - np.mean(selected_counts)) ** 2)
            coeff_det_r2 = 1 - ss_res / ss_tot
            flag = True
    except OptimizeWarning:
        return flag, coeff_det_r2, popt, perr
    except RuntimeError:
        return flag, coeff_det_r2, popt, perr
    except TypeError:
        return flag, coeff_det_r2, popt, perr
    except ValueError:
        return flag, coeff_det_r2, popt, perr

    return flag, coeff_det_r2, popt, perr


def gauss_fit_max(
    matrix: np.ndarray,
    tmp_labels: np.ndarray,
    bins: int | str,
    number_of_sigmas: float,
) -> StateUni | None:
    """
    Selection of the optimal interval and parameters in order to fit a state.

    Parameters
    ----------

    """
    mask = tmp_labels == 0
    flat_m = matrix[mask].flatten()

    try:
        kde = gaussian_kde(flat_m)
    except ValueError:
        return None

    if bins == "auto":
        binning = np.linspace(np.min(flat_m), np.max(flat_m), 100)
    else:
        binning = np.linspace(np.min(flat_m), np.max(flat_m), int(bins))
    counts = kde.evaluate(binning)

    gap = 3
    max_val = counts.max()
    max_ind = counts.argmax()

    min_id0 = np.max([max_ind - gap, 0])
    min_id1 = np.min([max_ind + gap, counts.size - 1])
    while min_id0 > 0 and counts[min_id0] > counts[min_id0 - 1]:
        min_id0 -= 1
    while min_id1 < counts.size - 1 and counts[min_id1] > counts[min_id1 + 1]:
        min_id1 += 1

    fit_param = [min_id0, min_id1, max_ind, flat_m.size]
    fit_data = [binning, counts]
    flag_min, r_2_min, popt_min, _ = perform_gauss_fit(
        fit_param, fit_data, "Min"
    )

    half_id0 = np.max([max_ind - gap, 0])
    half_id1 = np.min([max_ind + gap, counts.size - 1])
    while half_id0 > 0 and counts[half_id0] > max_val / 2:
        half_id0 -= 1
    while half_id1 < counts.size - 1 and counts[half_id1] > max_val / 2:
        half_id1 += 1

    fit_param = [half_id0, half_id1, max_ind, flat_m.size]
    fit_data = [binning, counts]
    flag_half, r_2_half, popt_half, _ = perform_gauss_fit(
        fit_param, fit_data, "Half"
    )

    r_2 = r_2_min
    if flag_min == 1 and flag_half == 0:
        popt = popt_min
    elif flag_min == 0 and flag_half == 1:
        popt = popt_half
        r_2 = r_2_half
    elif flag_min * flag_half == 1:
        if r_2_min >= r_2_half:
            popt = popt_min
        else:
            popt = popt_half
            r_2 = r_2_half
    else:
        return None

    state = StateUni(popt[0], popt[1], popt[2], r_2)
    state._build_boundaries(number_of_sigmas)

    return state


def find_stable_trj(
    matrix: np.ndarray,
    tmp_labels: np.ndarray,
    state: StateUni,
    delta_t: int,
    lim: int,
) -> Tuple[np.ndarray, float]:
    """
    Identification of windows contained in a certain state.

    Parameters
    ----------

    cl_ob : ClusteringObject1D
        The clustering object.

    state : StateUni
        The state we want to put windows in.

    tmp_labels : ndarray of shape (n_particles, n_windows)
        Contains the cluster labels of all the signal windows.

    lim : int
        The algorithm iteration.

    Returns
    -------

    m2_array : ndarray
        Array of still unclassified data points.

    window_fraction : float
        Fraction of windows classified in this state.

    env_0 : bool
        Indicates if there are still unclassified data points.
    """
    mask_unclassified = tmp_labels == 0
    mask_inf = matrix >= state.th_inf[0]
    mask_sup = matrix <= state.th_sup[0]
    mask = mask_unclassified & mask_inf & mask_sup

    mask_stable = np.zeros_like(matrix, dtype=bool)
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
    fraction = np.sum(mask_stable) / matrix.size

    return tmp_labels, fraction


def fit_local_maxima(
    cl_ob: ClusteringObject1D,
    m_clean: np.ndarray,
    par: Parameters,
    tmp_labels: np.ndarray,
    lim: int,
):
    """
    This functions takes care of particular cases where the
    data points on the tails of a Gaussian are not correctly assigned,
    creating weird sharp peaks in the histogram.

    Parameters
    ----------

    cl_ob : ClusteringObject1D

    m_clean : np.ndarray
        Input data for Gaussian fitting.

    par : Parameters
        Object containing parameters for fitting.

    tmp_labels : np.ndarray
        Labels indicating window classifications.

    tau_windw : int
        The time resolution of the analysis.

    lim : int
        Offset value for classifying stable windows.
    """
    flat_m = m_clean.flatten()

    kde = gaussian_kde(flat_m)
    if par.bins == "auto":
        bins = np.linspace(np.min(flat_m), np.max(flat_m), 100)
    else:
        bins = np.linspace(np.min(flat_m), np.max(flat_m), int(par.bins))
    counts = kde.evaluate(bins)

    gap = 3

    max_ind, _ = scipy.signal.find_peaks(counts)
    max_val = np.array([counts[i] for i in max_ind])

    for i, m_ind in enumerate(max_ind[:1]):
        min_id0 = np.max([m_ind - gap, 0])
        min_id1 = np.min([m_ind + gap, counts.size - 1])
        while min_id0 > 0 and counts[min_id0] > counts[min_id0 - 1]:
            min_id0 -= 1
        while (
            min_id1 < counts.size - 1 and counts[min_id1] > counts[min_id1 + 1]
        ):
            min_id1 += 1

        fit_param = [min_id0, min_id1, m_ind, flat_m.size]
        fit_data = [bins, counts]
        flag_min, r_2_min, popt_min, _ = perform_gauss_fit(
            fit_param, fit_data, "Min"
        )

        half_id0 = np.max([m_ind - gap, 0])
        half_id1 = np.min([m_ind + gap, counts.size - 1])
        while half_id0 > 0 and counts[half_id0] > max_val[i] / 2:
            half_id0 -= 1
        while half_id1 < counts.size - 1 and counts[half_id1] > max_val[i] / 2:
            half_id1 += 1

        fit_param = [half_id0, half_id1, m_ind, flat_m.size]
        fit_data = [bins, counts]
        flag_half, r_2_half, popt_half, _ = perform_gauss_fit(
            fit_param, fit_data, "Half"
        )

        r_2 = r_2_min
        if flag_min == 1 and flag_half == 0:
            popt = popt_min
        elif flag_min == 0 and flag_half == 1:
            popt = popt_half
            r_2 = r_2_half
        elif flag_min * flag_half == 1:
            if r_2_min >= r_2_half:
                popt = popt_min
            else:
                popt = popt_half
                r_2 = r_2_half
        else:
            continue

        state = StateUni(popt[0], popt[1], popt[2], r_2)
        state._build_boundaries(par.number_of_sigmas)

        m_clean = cl_ob.data.matrix

        mask_unclassified = tmp_labels < 0.5
        mask_inf = np.min(m_clean, axis=1) >= state.th_inf[0]
        mask_sup = np.max(m_clean, axis=1) <= state.th_sup[0]
        mask = mask_unclassified & mask_inf & mask_sup

        tmp_labels[mask] = lim + 1
        counter = np.sum(mask)

        mask_remaining = mask_unclassified & ~mask
        remaning_data = m_clean[mask_remaining]
        m2_array = np.array(remaning_data)

        if tmp_labels.size == 0:
            return None, None, None, None
        else:
            window_fraction = counter / tmp_labels.size

        one_last_state = True
        if len(m2_array) == 0:
            one_last_state = False

        return state, m2_array, window_fraction, one_last_state

    return None, None, None, None


def iterative_search(
    matrix: np.ndarray,
    delta_t: int,
    bins: int | str,
    number_of_sigmas: float,
) -> tuple[list[StateUni], NDArray[np.int64]]:
    """
    Iterative search for stable windows in the trajectory.

    Parameters
    ----------

    """
    tmp_labels = np.zeros(matrix.shape, dtype=int)
    tmp_states_list = []
    states_counter = 0

    while True:
        state = gauss_fit_max(
            matrix,
            tmp_labels,
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
        )
        if counter == 0.0:
            break

        # if counter == 0.0:
        #     state, m_next, counter = fit_local_maxima(
        #         m_copy,
        #         cl_ob.par,
        #         tmp_labels,
        #         states_counter,
        #     )

        #     if counter == 0 or state is None:
        #         break

        state.perc = counter
        tmp_states_list.append(state)
        states_counter += 1

    labels, state_list = relabel_states(tmp_labels, tmp_states_list)

    return state_list, labels - 1


def _main(
    matrix: np.ndarray,
    delta_t: int,
    bins: int | str,
    number_of_sigmas: float,
    max_area_overlap: float,
) -> tuple[list[StateUni], NDArray[np.int64]]:
    """
    Returns the clustering object with the analysis.

    Parameters
    ----------
    matrix : ndarray of shape (n_particles * n_windows, tau_window)
        The values of the signal for each particle at each frame.

    bins: Union[str, int] = "auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/
        numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigma : float = 2.0
        Sets the thresholds for classifing a signal window inside a state:
        the window is contained in the state if it is entirely contained
        inside number_of_sigma * state.sigms times from state.mean.

    Returns
    -------
    states_list : List[StateUni]
        The list of the identified states. Refer to the documentation of
        StateUni for accessing the information on the states.

    labels : ndarray of shape (n_particles * n_seq,)
        Cluster labels for each signal sequence. Unclassified points are given
        the label "-1".
    """
    tmp_state_list, tmp_labels = iterative_search(
        matrix,
        delta_t,
        bins,
        number_of_sigmas,
    )

    if len(tmp_state_list) > 0:
        state_list, labels = set_final_states(
            tmp_state_list,
            tmp_labels,
            max_area_overlap,
        )
    else:
        state_list = tmp_state_list
        labels = -np.ones(matrix.shape)

    return state_list, labels
