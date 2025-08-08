"""Auxiliary/discarded functions, not needed for the clustering."""

# Author: Becchi Matteo <bechmath@gmail.com>

import numpy as np
import scipy
from numpy.typing import NDArray


def reshape_from_nt(
    input_data: NDArray[np.float64],
    delta_t: int,
) -> NDArray[np.float64]:
    """
    Reshapes the input data from traditional to scikit format.

    Takes the array containing the univariate time-series data in the
    (n_particles, n_frames) format and reshapes it in the format required
    by scikit-learn (-1, delta_t).

    Parameters
    ----------

    input_data : np.ndarray of shape (n_particles, n_frames)
        The data to cluster in the traditional shape.

    delta_t : int
        Length of the signal sequence - the analysis time resolution.

    Returns
    -------

    reshaped_data : np.ndarray of shape (n_particles * n_seq, delta_t)
        The data to cluster in the scikit-required shape.

    Example
    -------

    .. testcode:: reshape_nt-test

        import numpy as np
        from tropea_clustering import helpers

        # Select time resolution
        delta_t = 10

        # Create random input data
        np.random.seed(1234)
        n_particles = 5
        n_frames = 1000

        input_data = np.random.rand(n_particles, n_frames)

        # Create input array with the scikit-required shape
        reshaped_input_data = helpers.reshape_from_nt(
            input_data, delta_t,
        )

    .. testcode:: reshape_nt-test
            :hide:

            assert reshaped_input_data.shape == (500, 10)
    """
    n_particles, n_frames = input_data.shape
    n_seq = n_frames // delta_t
    frames_in_excess = n_frames - n_seq * delta_t

    if frames_in_excess > 0:
        reshaped_data = np.reshape(
            input_data[:, :-frames_in_excess],
            (n_particles * n_seq, delta_t),
        )
    else:
        reshaped_data = np.reshape(
            input_data,
            (n_particles * n_seq, delta_t),
        )

    return reshaped_data


def reshape_from_dnt(
    input_data: NDArray[np.float64],
    delta_t: int,
) -> NDArray[np.float64]:
    """
    Reshapes the input data from traditional from scikit format.

    Takes the array containing the univariate time-series data in the
    (n_dims, n_particles, n_frames) format and reshapes it in the format
    required by scikit-learn (-1, delta_t * n_dims).

    Parameters
    ----------

    input_data : np.ndarray of shape (n_dims, n_particles, n_frames)
        The data to cluster in the traditional shape.

    delta_t : int
        Length of the signal sequence - the analysis time resolution.

    Returns
    -------

    reshaped_data : np.ndarray of shape (n_particles * n_seq, delta_t * n_dim)
        The data to cluster in the scikit-required shape.

    Example
    -------

    .. testcode:: reshape_dnt-test

        import numpy as np
        from tropea_clustering import helpers

        # Select time resolution
        delta_t = 5

        # Create random input data
        np.random.seed(1234)
        n_dims = 2
        n_particles = 5
        n_frames = 1000

        input_data = np.random.rand(n_dims, n_particles, n_frames)

        # Create input array with the scikit-required shape
        reshaped_input_data = helpers.reshape_from_dnt(
            input_data, delta_t,
        )

    .. testcode:: reshape_dnt-test
            :hide:

            assert reshaped_input_data.shape == (1000, 10)
    """
    _, n_particles, n_frames = input_data.shape
    n_seq = n_frames // delta_t
    frames_in_excess = n_frames - n_seq * delta_t

    if frames_in_excess > 0:
        reshaped_data = np.reshape(
            input_data[:, :, :-frames_in_excess], (n_particles * n_seq, -1)
        )
    else:
        reshaped_data = np.reshape(input_data, (n_particles * n_seq, -1))

    return reshaped_data


def butter_lowpass_filter(x: np.ndarray, cutoff: float, fs: float, order: int):
    nyq = 0.5
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
    y = scipy.signal.filtfilt(b, a, x)
    return y


def Savgol_filter(m: np.ndarray, window: int):
    # Step 1: Set the polynomial order for the Savitzky-Golay filter.
    poly_order = 2

    # Step 2: Apply the Savitzky-Golay filter to each row (x) in the input data matrix 'm'.
    # The result is stored in a temporary array 'tmp'.
    # 'window' is the window size for the filter.
    tmp = np.array(
        [scipy.signal.savgol_filter(x, window, poly_order) for x in m]
    )

    # Step 3: Since the Savitzky-Golay filter operates on a sliding window,
    # it introduces edge artifacts at the beginning and end of each row.
    # To remove these artifacts, the temporary array 'tmp' is sliced to remove the unwanted edges.
    # The amount of removal on each side is half of the 'window' value, converted to an integer.
    return tmp[:, int(window / 2) : -int(window / 2)]


def normalize_array(x: np.ndarray):
    # Step 1: Calculate the mean value and the standard deviation of the input array 'x'.
    mean = np.mean(x)
    stddev = np.std(x)

    # Step 2: Create a temporary array 'tmp' containing the normalized version of 'x'.
    # To normalize, subtract the mean value from each element of 'x'
    # and then divide by the standard deviation.
    # This centers the data around zero (mean) and scales it based on the standard deviation.
    tmp = (x - mean) / stddev

    # Step 3: Return the normalized array 'tmp',
    # along with the calculated mean and standard deviation.
    # The returned values can be useful for further processing
    # or to revert the normalization if needed.
    return tmp, mean, stddev


def sigmoidal(x: float, a: float, b: float, alpha: float):
    return b + a / (1 + np.exp(x * alpha))


def gaussian_2d(
    r_points: np.ndarray,
    x_mean: float,
    y_mean: float,
    sigmax: float,
    sigmay: float,
    area: float,
):
    """Compute the 2D Gaussian function values at given radial points 'r_points'.

    Args:
    - r_points (np.ndarray): Array of radial points in a 2D space.
    - x_mean (float): Mean value along the x-axis of the Gaussian function.
    - y_mean (float): Mean value along the y-axis of the Gaussian function.
    - sigmax (float): Standard deviation along the x-axis of the Gaussian function.
    - sigmay (float): Standard deviation along the y-axis of the Gaussian function.
    - area (float): Total area under the 2D Gaussian curve.

    Returns:
    - np.ndarray: 2D Gaussian function values computed at the radial points.

    This function calculates the values of a 2D Gaussian function at given radial points 'r_points'
    centered around the provided means ('x_mean' and 'y_mean') and standard deviations
    ('sigmax' and 'sigmay'). The 'area' parameter represents the total area
    under the 2D Gaussian curve. It returns an array of 2D Gaussian function values
    computed at the input radial points 'r_points'.
    """

    r_points[0] -= x_mean
    r_points[1] -= y_mean
    arg = (r_points[0] / sigmax) ** 2 + (r_points[1] / sigmay) ** 2
    normalization = np.pi * sigmax * sigmay
    gauss = np.exp(-arg) * area / normalization
    return gauss.ravel()


def gaussian_full(
    r: np.ndarray,
    mx: float,
    my: float,
    sigmax: float,
    sigmay: float,
    sigmaxy: float,
    area: float,
):
    # "m" is the Gaussians' mean value (2d array)
    # "sigma" is the Gaussians' standard deviation matrix
    # "area" is the Gaussian area
    r[0] -= mx
    r[1] -= my
    arg = (
        (r[0] / sigmax) ** 2
        + (r[1] / sigmay) ** 2
        + 2 * r[0] * r[1] / sigmaxy**2
    )
    norm = (
        np.pi
        * sigmax
        * sigmay
        / np.sqrt(1 - (sigmax * sigmay / sigmaxy**2) ** 2)
    )
    gauss = np.exp(-arg) * area / norm
    return gauss.ravel()
