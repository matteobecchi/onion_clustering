"""
Contains the classes used for storing parameters and system states.
"""

from typing import Union

import numpy as np


class StateUni:
    """
    Represents a unidimensional state as a Gaussian.

    All the parameters and information on the Gaussian states corresponding
    to the different clusters are stored within this class. The attributes can
    be acessed using the get_attributes() method.

    Parameters
    ----------

    mean : float
        Mean of the Gaussian.

    sigma : float
        Rescaled standard deviation of the Gaussian.

    area : float
        Area below the Gaussian.

    r_2 : float
        Coefficient of determination of the Gaussian fit.

    Attributes
    ----------

    peak : float
        Maximum value of the Gaussian.

    perc : float
        Fraction of data points classified in the state.

    th_inf : float
        Lower threshold of the state. Considering the Gaussian states oreder
        with increasing values of the mean, this is the intercsection point
        (if exists) with the Gaussian before. If there is no intersection, it
        is the weighted average between the two means. The two cases are
        distinguished by the value of th_inf[1], which is "0" in the first
        case, "1" in the second. The actual threshold value is stored in
        th_inf[0].

    th_sup : float
        Upper thrashold of the state. Considering the Gaussian states oreder
        with increasing values of the mean, this is the intercsection point
        (if exists) with the Gaussian after. If there is no intersection, it
        is the weighted average between the two means. The two cases are
        distinguished by the value of th_sup[1], which is "0" in the first
        case, "1" in the second. The actual threshold value is stored in
        th_sup[0].
    """

    def __init__(self, mean: float, sigma: float, area: float, r_2: float):
        self._mean = mean
        self._sigma = sigma
        self._area = area
        self._peak = area / sigma / np.sqrt(np.pi)
        self._r_2 = r_2
        self._perc = 0.0
        self._th_inf = [mean - 2.0 * sigma, -1]
        self._th_sup = [mean + 2.0 * sigma, -1]

    def _build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds to classify the data windows inside the state.

        Parameters
        ----------

        number of sigmas : float
            How many sigmas the thresholds are far from the mean.
        """
        self._th_inf = [self._mean - number_of_sigmas * self._sigma, -1]
        self._th_sup = [self._mean + number_of_sigmas * self._sigma, -1]

    def get_attributes(self):
        """
        Returns a dictionary containing the attributes of the state.

        The attributes "th_inf" and "th_sup" are returned as a single ndarray
        with the label "th".

        Returns
        -------
        attr_list : dict
        """
        attr_list = {
            "mean": self._mean,
            "sigma": self._sigma,
            "area": self._area,
            "peak": self._peak,
            "r_2": self._r_2,
            "perc": self._perc,
            "th": [self._th_inf, self._th_sup],
        }
        return attr_list


class StateMulti:
    """
    Represents a multifimensional state as a factorized Gaussian.

    All the parameters and information on the factorized Gaussian states
    corresponding to the different clusters are stored within this class.
    The attributes can be acessed using the get_attributes() method.

    Parameters
    ----------

    mean : np.ndarray of shape (dim,)
        Mean of the Gaussians.

    sigma : np.ndarray of shape (dim,)
        Rescaled standard deviation of the Gaussians.

    area : np.ndarray of shape (dim,)
        Area below the Gaussians.

    r_2 : float
        Coefficient of determination of the Gaussian fit.

    Attributes
    ----------

    perc : float
        Fraction of data points classified in this state.

    axis : ndarray of shape (dim,)
        The thresholds of the state. It contains the axis of the ellipsoid
        given by the rescaled sigmas of the factorized Gaussian states,
        multiplied by "number of sigmas".
    """

    def __init__(
        self, mean: np.ndarray, sigma: np.ndarray, area: np.ndarray, r_2: float
    ):
        self._mean = mean
        self._sigma = sigma
        self._area = area
        self._r_2 = r_2
        self._perc = 0.0
        self._axis = 2.0 * sigma

    def _build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds to classify the data windows inside the state.

        Parameters
        ----------

        number of sigmas : float
            How many sigmas the thresholds are far from the mean.
        """
        self._axis = number_of_sigmas * self._sigma  # Axes of the state

    def get_attributes(self):
        """
        Returns a dictionary containing the attributes of the state.

        Returns
        -------
        attr_list : dict
        """
        attr_list = {
            "mean": self._mean,
            "sigma": self._sigma,
            "area": self._area,
            "r_2": self._r_2,
            "perc": self._perc,
            "axis": self._axis,
        }
        return attr_list


class UniData:
    """
    The input univariate signals to cluster.

    Parameters
    ----------

    matrix : ndarray of shape (n_particles, n_frames)
        The values of the signal for each particle at each frame.

    Attributes
    ----------

    number_of_particles : int
        The number of particles in the system.

    num_of_steps : int
        The number of frames in the system.

    range : ndarray of shape (2,)
        Min and max of the signals.

    labels : ndarray of shape (n_particles, n_frames)
        The cluster labels.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.num_of_particles = self.matrix.shape[0]
        self.num_of_steps = self.matrix.shape[1]
        self.labels = np.array([])
        if matrix.size > 0:
            self.range = np.array([np.min(self.matrix), np.max(self.matrix)])


class MultiData:
    """
    The input mutivariate signals to cluster.

    Parameters
    ----------

    matrix : ndarray of shape (dims, n_particles, n_frames)
        The values of the signal for each particle at each frame.

    Attributes
    ----------

    dims : int
        The dimension of the space of the signals.

    number_of_particles : int
        The number of particles in the system.

    num_of_steps : int
        The number of frames in the system.

    range : ndarray of shape (dim, 2)
        Min and max of the signals along each axes.

    matrix : ndarray of shape (n_particles, n_frames, dims)
        The values of the signal for each particle at each frame.

    labels : ndarray of shape (n_particles, n_frames)
        The cluster labels.
    """

    def __init__(self, matrix: np.ndarray):
        self.dims = matrix.shape[0]
        self.num_of_particles = matrix.shape[1]
        self.num_of_steps = matrix.shape[2]
        self.range = np.array(
            [[np.min(data), np.max(data)] for data in matrix]
        )

        self.matrix: np.ndarray = np.transpose(matrix, axes=(1, 2, 0))
        self.labels = np.array([])


class Parameters:
    """
    Contains the set of parameters for the specific analysis.

    Parameters
    ----------

    tau_w : int
        The time resolution for the clustering, corresponding to the length
        of the windows in which the time-series are segmented.

    bins: Union[str, int]
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/
        numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).
    """

    def __init__(
        self,
        tau_window: int,
        bins: Union[int, str],
        number_of_sigmas: float,
    ):
        self.tau_w = tau_window
        self.bins = bins
        self.number_of_sigmas = number_of_sigmas