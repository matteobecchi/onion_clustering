"""onion-clustering for multivariate time-series."""

# Author: Becchi Matteo <bechmath@gmail.com>
# Reference: https://www.pnas.org/doi/abs/10.1073/pnas.2403771121

import numpy as np
from numpy.typing import NDArray

from tropea_clustering._internal.onion_smooth.main_2d import StateMulti
from tropea_clustering._internal.onion_smooth.main_2d import (
    _main as _onion_inner,
)


def onion_multi_smooth(
    X: NDArray[np.float64],
    delta_t: int,
    bins: str | int = "auto",
    number_of_sigmas: float = 3.0,
) -> tuple[list[StateMulti], NDArray[np.int64]]:
    """
    Performs onion clustering on the data array 'X'.

    Returns an array of integer labels, one for each frame.
    Unclassified frames are labelled "-1".

    .. note::
        This function is currently in beta testing. The output could change
        in the future. Use with caution.

    Parameters
    ----------
    X : ndarray of shape (n_particles, n_frames, n_features)
        The time-series data to cluster.

    delta_t : int
        The minimum lifetime required for the clusters. Also referred to as
        the "time resolution" of the clustering analysis.

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
    states_list : List[StateMulti]
        The list of the identified states.Refer to the documentation of
        StateMulti for accessing the information on the states.

    labels : ndarray of shape (n_particles, n_frames)
        Cluster labels for each frame. Unclassified points are given
        the label "-1".

    Example
    -------

    .. testcode:: onion_multi_smooth-test

        import numpy as np
        from tropea_clustering import onion_multi_smooth

        # Select time resolution
        delta_t = 2

        # Create random input data
        np.random.seed(1234)
        n_features = 2
        n_particles = 5
        n_steps = 1000

        input_data = np.random.rand(n_particles, n_steps, n_features)

        # Run Onion Clustering
        state_list, labels = onion_multi_smooth(input_data, delta_t)

    .. testcode:: onion_multi_smooth-test
            :hide:

            assert np.isclose(state_list[0].mean[0], 0.4791087814511593)
    """

    est = OnionMultiSmooth(
        delta_t=delta_t,
        bins=bins,
        number_of_sigmas=number_of_sigmas,
    )
    est.fit(X)

    return est.state_list_, est.labels_


class OnionMultiSmooth:
    """
    Performs onion clustering on a data array.

    Returns an array of integer labels, one for each frame.
    Unclassified frames are labelled "-1".

    .. note::
        This class is currently in beta testing. Its operations could change
        in the future. Use with caution.

    Parameters
    ----------
    delta_t : int
        The minimum lifetime required for the clusters. Also referred to as
        the "time resolution" of the clustering analysis.

    bins : int, default="auto"
        The number of bins used for the construction of the histograms.
        Can be an integer value, or "auto".
        If "auto", the default of numpy.histogram_bin_edges is used
        (see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges).

    number_of_sigmas : float, default=3.0
        Sets the thresholds for classifing a signal sequence inside a state:
        the sequence is contained in the state if it is entirely contained
        inside number_of_sigma * state.sigms times from state.mean.

    Attributes
    ----------
    state_list_ : List[StateMulti]
        The list of the identified states. Refer to the documentation of
        StateMulti for accessing the information on the states.

    labels_: ndarray of shape (n_particles, n_frames)
        Cluster labels for each frame. Unclassified points are given
        the label "-1".

    Example
    -------

    .. testcode:: OnionMultiSmooth-test

        import numpy as np
        from tropea_clustering import OnionMultiSmooth

        # Select time resolution
        delta_t = 2

        # Create random input data
        np.random.seed(1234)
        n_features = 2
        n_particles = 5
        n_steps = 1000

        input_data = np.random.rand(n_particles, n_steps, n_features)

        # Run Onion Clustering
        clusterer = OnionMultiSmooth(delta_t)
        clust_params = {"bins": 100, "number_of_sigmas": 2.0}
        clusterer.set_params(**clust_params)
        clusterer.fit(input_data)

    .. testcode:: OnionMultiSmooth-test
            :hide:

            assert np.isclose(
                clusterer.state_list_[0].mean[0], 0.6257886444256409)
    """

    def __init__(
        self,
        delta_t: int,
        bins: str | int = "auto",
        number_of_sigmas: float = 3.0,
    ):
        self.delta_t = delta_t
        self.bins = bins
        self.number_of_sigmas = number_of_sigmas

    def fit(self, X, y=None):
        """Performs onion clustering on the data array 'X'.

        Parameters
        ----------
        X : ndarray of shape (n_particles, n_frames, n_features)
            The time-series data to cluster.

        Returns
        -------
        self : object
            A fitted instance of self.
        """
        if X.ndim != 3:
            raise ValueError("Expected 3-dimensional input data.")

        if X.shape[0] == 0:
            raise ValueError("Empty dataset.")

        if X.shape[1] <= 1:
            raise ValueError("n_frames = 1.")

        # Check for complex input
        if not (
            np.issubdtype(X.dtype, np.floating)
            or np.issubdtype(X.dtype, np.integer)
        ):
            raise ValueError("Complex data not supported.")

        X = X.copy()  # copy to avoid in-place modification

        self.state_list_, self.labels_ = _onion_inner(
            X,
            self.delta_t,
            self.bins,
            self.number_of_sigmas,
        )

        return self

    def fit_predict(self, X, y=None):
        """Computes clusters on the data array 'X' and returns labels.

        Parameters
        ----------
        X : ndarray of shape (n_particles, n_frames, n_features)
            The time-series data to cluster.

        Returns
        -------
        labels_: ndarray of shape (n_particles, n_frames)
            Cluster labels for each frame. Unclassified points are given
            the label "-1".
        """
        return self.fit(X).labels_

    def get_params(self, deep=True):
        return {
            "delta_t": self.delta_t,
            "bins": self.bins,
            "number_of_sigmas": self.number_of_sigmas,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
