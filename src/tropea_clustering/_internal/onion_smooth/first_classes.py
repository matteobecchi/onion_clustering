"""
Contains the classes used for storing parameters and system states.
"""

# Author: Becchi Matteo <bechmath@gmail.com>

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
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

    th_inf : ndarray of shape (2,)
        _th_inf[0] stores the lower threshold of the state. Considering the
        Gaussian states oreder with increasing values of the mean, this is the
        intercsection point (if exists) with the Gaussian before. If there is
        no intersection, it is the weighted average between the two means.
        The two cases are distinguished by the value of _th_inf[1], which is
        "0" in the first case, "1" in the second.

    th_sup : ndarray of shape (2,)
        _th_sup[0] stores the upper threshold of the state. Considering the
        Gaussian states oreder with increasing values of the mean, this is the
        intercsection point (if exists) with the Gaussian after. If there is
        no intersection, it is the weighted average between the two means.
        The two cases are distinguished by the value of _th_sup[1], which is
        "0" in the first case, "1" in the second.
    """

    mean: float
    sigma: float
    area: float
    r_2: float
    perc: float = 0.0
    peak: float = field(init=False)
    th_inf: np.ndarray = field(init=False)
    th_sup: np.ndarray = field(init=False)

    def __post_init__(self):
        self.peak = self.area / self.sigma / np.sqrt(np.pi)
        self.th_inf = [self.mean - 2.0 * self.sigma, -1]
        self.th_sup = [self.mean + 2.0 * self.sigma, -1]

    def _build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds to classify the data windows inside the state.

        Parameters
        ----------

        number of sigmas : float
            How many sigmas the thresholds are far from the mean.
        """
        self.th_inf = np.array([self.mean - number_of_sigmas * self.sigma, -1])
        self.th_sup = np.array([self.mean + number_of_sigmas * self.sigma, -1])

    def get_attributes(self):
        """
        Returns a dictionary containing the attributes of the state.

        The attributes "th_inf" and "th_sup" are returned as a single ndarray
        with the label "th".

        Returns
        -------
        attr_list : dict
        """
        attr_list = asdict(self)
        return attr_list


@dataclass
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

    covariance : np.ndarray of shape (dim, dim)
        Covariance matrix of the Gaussians.

    log_likelihood : float
        log_likelihood of the data under the fitted Gaussian.

    Attributes
    ----------

    perc : float
        Fraction of data points classified in this state.

    axis : ndarray of shape (dim,)
        The thresholds of the state. It contains the axis of the ellipsoid
        given by the rescaled sigmas of the factorized Gaussian states,
        multiplied by "number of sigmas".
    """

    mean: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    perc: float = 0.0
    axis: np.ndarray = field(init=False)

    def __post_init__(self):
        self.axis = 2.0 * self.covariance

    def _build_boundaries(self, number_of_sigmas: float):
        """
        Sets the thresholds to classify the data windows inside the state.

        Parameters
        ----------

        number of sigmas : float
            How many sigmas the thresholds are far from the mean.
        """
        self.axis = number_of_sigmas * self.covariance  # Axes of the state

    def get_attributes(self):
        """
        Returns a dictionary containing the attributes of the state.

        Returns
        -------
        attr_list : dict
        """
        attr_list = asdict(self)
        return attr_list
