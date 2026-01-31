"""Pytest for Onion class."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tropea_clustering import Onion, onion_clustering

# ---------------- Fixtures ----------------


@pytest.fixture(scope="module")
def input_data_2d() -> np.ndarray:
    np.random.seed(42)
    mu1 = np.array([0.0, 0.0])
    cov1 = np.array([[0.05, 0.02], [0.02, 0.02]])
    mu2 = np.array([1.0, 1.0])
    cov2 = np.array([[0.1, -0.03], [-0.03, 0.05]])
    list_data_2d = []
    for _ in range(100):
        points1 = np.random.multivariate_normal(mu1, cov1, size=500)
        points2 = np.random.multivariate_normal(mu2, cov2, size=500)
        time_series = np.vstack([points1, points2])
        list_data_2d.append(time_series)
    return np.array(list_data_2d)


# ---------------- Tests ----------------


def test_onion(input_data_2d: np.ndarray):
    delta_t = 10

    # Test class interface
    on_cl = Onion(bins=50)
    labels = on_cl.fit_predict(input_data_2d, delta_t)

    # Test functional interface
    _ = onion_clustering(input_data_2d, delta_t)

    # Check clustering output
    this_dir = Path(__file__).parent
    expected = np.load(this_dir / "output_multi_smooth/labels.npy")
    mask_0 = expected == 0
    mask_1 = expected == 1
    expected[mask_1] = 0
    expected[mask_0] = 1
    assert_array_equal(labels, expected)


def test_wrong_input():
    on_cl = Onion()

    # input_data = np.ones((10, 10))  # 2D array
    # with pytest.raises(ValueError, match="Expected 3-dimensional input data."):
    #     on_cl.fit(input_data, delta_t=10)

    input_data1 = np.empty((0, 5, 5))  # empty array
    with pytest.raises(ValueError, match="Empty dataset."):
        on_cl.fit(input_data1, delta_t=10)

    input_data1 = np.zeros((100, 1, 2))  # just one frame
    with pytest.raises(ValueError, match="n_frames = 1."):
        on_cl.fit(input_data1, delta_t=10)

    input_data = np.random.rand(3, 4, 2) + 1j * np.random.rand(3, 4, 2)
    with pytest.raises(ValueError, match="Complex data not supported."):
        on_cl.fit(input_data, delta_t=10)
