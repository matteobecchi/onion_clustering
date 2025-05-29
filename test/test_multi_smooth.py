"""Pytest for onion_multi_smooth and OnionMultiSmooth."""

import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tropea_clustering import OnionMultiSmooth, onion_multi_smooth


def test_onion_multi_smooth():
    rng = np.random.default_rng(12345)
    input_data_2d = np.array(
        [
            np.concatenate(
                (
                    rng.normal(0.0, 0.1, (500, 2)),
                    rng.normal(1.0, 0.1, (500, 2)),
                )
            )
            for _ in range(100)
        ]
    )
    _ = np.array(
        [
            np.concatenate(
                (
                    rng.normal(0.0, 0.1, (500, 3)),
                    rng.normal(1.0, 0.1, (500, 3)),
                )
            )
            for _ in range(100)
        ]
    )

    delta_t = 10

    # Test class interface
    on_cl = OnionMultiSmooth(delta_t)
    tmp_params = {"bins": 50, "number_of_sigmas": 2.0}
    on_cl.set_params(**tmp_params)
    _ = on_cl.get_params()
    _ = on_cl.fit_predict(input_data_2d)

    # Test functional interface
    state_list, labels = onion_multi_smooth(input_data_2d, delta_t)
    _ = state_list[0].get_attributes()

    # Check clustering output
    this_dir = os.path.dirname(__file__)
    expected = np.load(
        os.path.join(this_dir, "output_multi_smooth", "labels.npy")
    )
    assert_array_equal(labels, expected)

    # # Test if also the 3D case works
    # state_list, labels = onion_multi(input_data_3d, delta_t)
    # expected_output_path = original_dir / "output_multi/labels_3D.npy"
    # expected_output = np.load(expected_output_path)
    # assert np.allclose(expected_output, labels)

    # Test wrong input
    input_data = np.ones((10, 10))  # 2D array
    with pytest.raises(ValueError, match="Expected 3-dimensional input data."):
        on_cl.fit(input_data)

    input_data1 = np.empty((0, 5, 5))  # empty array
    with pytest.raises(ValueError, match="Empty dataset."):
        on_cl.fit(input_data1)

    input_data1 = np.zeros((100, 1, 2))  # just one frame
    with pytest.raises(ValueError, match="n_frames = 1."):
        on_cl.fit(input_data1)

    input_data = np.random.rand(3, 4, 2) + 1j * np.random.rand(3, 4, 2)
    with pytest.raises(ValueError, match="Complex data not supported."):
        on_cl.fit(input_data)
