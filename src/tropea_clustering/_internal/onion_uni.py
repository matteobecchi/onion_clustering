"""onion-clustering for univariate time-series."""

# Author: Becchi Matteo <bechmath@gmail.com>
# Reference: https://www.pnas.org/doi/abs/10.1073/pnas.2403771121

from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data

from tropea_clustering._internal.main import StateUni
from tropea_clustering._internal.main import _main as _onion_inner


def onion_uni(
    X: NDArray[np.float64],
    bins: Union[str, int] = "auto",
    number_of_sigmas: float = 2.0,
) -> tuple[list[StateUni], NDArray[np.int64]]:
    """
    Performs onion clustering on the data array 'X'.

    Returns an array of integer labels, one for each signal sequence.
    Unclassified sequences are labelled "-1".

    Parameters
    ----------
    X : ndarray of shape (n_particles * n_seq, delta_t)
        The data to cluster. Each signal sequence is considered as a
        single data point.

    bins : int, default="auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigmas : float, default=2.0
        Sets the thresholds for classifing a signal sequence inside a state:
        the sequence is contained in the state if it is entirely contained
        inside number_of_sigmas * state.sigmas times from state.mean.

    Returns
    -------
    states_list : List[StateUni]
        The list of the identified states. Refer to the documentation of
        StateUni for accessing the information on the states.

    labels : ndarray of shape (n_particles * n_seq,)
        Cluster labels for each signal sequence. Unclassified points are given
        the label "-1".

    Example
    -------

    .. testcode:: onionuni-test

        import numpy as np
        from tropea_clustering import onion_uni, helpers

        # Select time resolution
        delta_t = 5

        # Create random input data
        np.random.seed(1234)
        n_particles = 5
        n_steps = 1000

        input_data = np.random.rand(n_particles, n_steps)

        # Create input array with the correct shape
        reshaped_input_data = helpers.reshape_from_nt(
            input_data, delta_t,
        )

        # Run Onion Clustering
        state_list, labels = onion_uni(reshaped_input_data)

    .. testcode:: onionuni-test
            :hide:

            assert np.isclose(state_list[0].mean, 0.5789299753284055)
    """

    est = OnionUni(
        bins=bins,
        number_of_sigmas=number_of_sigmas,
    )
    est.fit(X)

    return est.state_list_, est.labels_


class OnionUni(BaseEstimator, ClusterMixin):
    """
    Performs onion clustering on a data array.

    Returns an array of integer labels, one for each signal sequence.
    Unclassified sequences are labelled "-1".

    Parameters
    ----------
    bins : int, default="auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigmas : float, default=2.0
        Sets the thresholds for classifing a signal sequence inside a state:
        the sequence is contained in the state if it is entirely contained
        inside number_of_sigmas * state.sigmas times from state.mean.

    Attributes
    ----------
    state_list_ : List[StateUni]
        List of the identified states. Refer to the documentation of
        StateUni for accessing the information on the states.

    labels_: ndarray of shape (n_particles * n_seq,)
        Cluster labels for signal sequence. Unclassified points are given
        the label "-1".

    Example
    -------

    .. testcode:: OnionUni-test

        import numpy as np
        from tropea_clustering import OnionUni, helpers

        # Select time resolution
        delta_t = 5

        # Create random input data
        np.random.seed(1234)
        n_particles = 5
        n_steps = 1000

        input_data = np.random.rand(n_particles, n_steps)

        # Create input array with the correct shape
        reshaped_input_data = helpers.reshape_from_nt(
            input_data, delta_t,
        )

        # Run Onion Clustering
        clusterer = OnionUni()
        clust_params = {"bins": 100, "number_of_sigmas": 2.0}
        clusterer.set_params(**clust_params)
        clusterer.fit(reshaped_input_data)

    .. testcode:: OnionUni-test
            :hide:

            assert np.isclose(
                clusterer.state_list_[0].mean, 0.5789299753284055)
    """

    def __init__(
        self,
        bins: Union[str, int] = "auto",
        number_of_sigmas: float = 2.0,
    ):
        self.bins = bins
        self.number_of_sigmas = number_of_sigmas

    def fit(self, X, y=None):
        """Performs onion clustering on the data array 'X'.

        Parameters
        ----------
        X : ndarray of shape (n_particles * n_seq, delta_t)
            The data to cluster. Each signal sequence is considered as a
            single data point.

        Returns
        -------
        self : object
            A fitted instance of self.
        """
        X = validate_data(self, X=X, y=y, accept_sparse=False)

        if X.ndim != 2:
            raise ValueError("Expected 2-dimensional input data.")

        if X.shape[0] <= 1:
            raise ValueError("n_samples = 1")

        if X.shape[1] <= 1:
            raise ValueError("n_features = 1")

        # Check for complex input
        if not (
            np.issubdtype(X.dtype, np.floating)
            or np.issubdtype(X.dtype, np.integer)
        ):
            raise ValueError("Complex data not supported")

        X = X.copy()  # copy to avoid in-place modification

        cl_ob = _onion_inner(
            X,
            self.bins,
            self.number_of_sigmas,
        )

        self.state_list_ = cl_ob.state_list
        self.labels_ = cl_ob.data.labels

        return self

    def fit_predict(self, X, y=None):
        """Computes clusters on the data array 'X' and returns labels.

        Parameters
        ----------
        X : ndarray of shape (n_particles * n_seq, delta_t)
            The data to cluster. Each signal sequence is considered as a
            single data point.

        Returns
        -------
        labels_: ndarray of shape (n_particles * n_seq,)
            Cluster labels for signal sequence. Unclassified points are given
            the label "-1".
        """
        return self.fit(X).labels_

    def get_params(self, deep=True):
        return {
            "bins": self.bins,
            "number_of_sigmas": self.number_of_sigmas,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
