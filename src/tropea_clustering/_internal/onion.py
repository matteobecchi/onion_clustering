"""Main clustering object for Onion clustering."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tropea_clustering._internal.classes import OnionData, OnionParams
from tropea_clustering._internal.main import perform_onion_clustering


def onion_clustering(
    X: NDArray[np.float64],
    delta_t: int,
    bins: Literal["auto"] | int = "auto",
    number_of_sigmas: float = 3.0,
    max_area_overlap: float = 0.8,
) -> NDArray[np.int64]:
    """
    Performs onion clustering on the data array 'X'.

    Returns an array of integer labels, one for each frame.
    Unclassified frames are labelled "-1".

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

    max_area_overlap : float, default=0.8
        Thresold to consider two Gaussian states overlapping, and thus merge
        them together.

    Returns
    -------
    labels : ndarray of shape (n_particles, n_frames)
        Cluster labels for each frame. Unclassified points are given
        the label "-1".

    Example
    -------

    .. testcode:: onion-test

        import numpy as np
        from tropea_clustering import onion_clustering

        # Select time resolution
        delta_t = 2

        # Create random input data
        np.random.seed(1234)
        n_features = 2
        n_particles = 5
        n_steps = 1000

        input_data = np.random.rand(n_particles, n_steps, n_features)

        # Run Onion Clustering
        labels = onion_clustering(input_data, delta_t)

    .. testcode:: onion-test
            :hide:

            assert labels[0][0] == -1
    """

    est = Onion(
        bins=bins,
        number_of_sigmas=number_of_sigmas,
        max_area_overlap=max_area_overlap,
    )
    est.fit(X, delta_t=delta_t)

    return est.labels


class Onion:
    """
    Performs onion clustering on a data array.

    Returns an array of integer labels, one for each frame.
    Unclassified frames are labelled "-1".

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

    max_area_overlap : float, default=0.8
        Thresold to consider two Gaussian states overlapping, and thus merge
        them together.

    Attributes
    ----------
    state_list : List[dict]
        The list of the identified states.

    labels: ndarray of shape (n_particles, n_frames)
        Cluster labels for each frame. Unclassified points are given
        the label "-1".

    Example
    -------

    .. testcode:: Onion-test

        import numpy as np
        from tropea_clustering import Onion

        # Select time resolution
        delta_t = 2

        # Create random input data
        np.random.seed(1234)
        n_features = 2
        n_particles = 5
        n_steps = 1000

        input_data = np.random.rand(n_particles, n_steps, n_features)

        # Run Onion Clustering
        clust_params = {"bins": 100, "number_of_sigmas": 2.0}
        clusterer = Onion(**clust_params)
        clusterer.fit(input_data, delta_t)

    .. testcode:: Onion-test
            :hide:

            assert clusterer.labels[0][0] == -1
    """

    def __init__(
        self,
        bins: Literal["auto"] | int = "auto",
        number_of_sigmas: float = 3.0,
        max_area_overlap: float = 0.8,
    ):
        self.params = OnionParams(bins, number_of_sigmas, max_area_overlap)

    def fit(self, X, delta_t: int):
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
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("Expected 2- or 3-dimensional input data.")

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

        self.state_list, self.labels = perform_onion_clustering(
            OnionData(X),
            delta_t,
            self.params,
        )

        return self

    def fit_predict(self, X, delta_t):
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
        return self.fit(X, delta_t).labels
