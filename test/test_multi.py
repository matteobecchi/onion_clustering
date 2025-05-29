"""Pytest for onion_multi and OnionMulti."""

import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tropea_clustering import OnionMulti, helpers, onion_multi


def test_onion_multi():
    n_particles = 5
    n_steps = 1000
    delta_t = 10

    rng = np.random.default_rng(12345)
    random_walk_x = []
    random_walk_y = []
    for _ in range(n_particles):
        tmp_x, tmp_y = [0.0], [0.0]
        for _ in range(n_steps - 1):
            d_x = rng.normal()
            x_new = tmp_x[-1] + d_x
            tmp_x.append(x_new)
            d_y = rng.normal()
            y_new = tmp_y[-1] + d_y
            tmp_y.append(y_new)
        random_walk_x.append(tmp_x)
        random_walk_y.append(tmp_y)
    input_data = np.array([random_walk_x, random_walk_y])

    reshaped_input_data = helpers.reshape_from_dnt(input_data, delta_t)

    # Test class interface
    tmp = OnionMulti()
    tmp_params = {"bins": 50, "number_of_sigmas": 2.0}
    tmp.set_params(**tmp_params)
    _ = tmp.get_params()
    tmp.fit_predict(reshaped_input_data)

    # Test functional interface
    state_list, labels = onion_multi(reshaped_input_data)
    _ = state_list[0].get_attributes()

    # Check clustering output
    this_dir = os.path.dirname(__file__)
    expected = np.load(os.path.join(this_dir, "output_multi", "labels.npy"))
    assert_array_equal(labels, expected)

    # Check also 3D case
    random_input = rng.random((3, 100, 200))
    reshaped_input_data = helpers.reshape_from_dnt(random_input, delta_t)
    state_list, labels = onion_multi(reshaped_input_data, ndims=3)

    expected = np.load(os.path.join(this_dir, "output_multi", "labels_3D.npy"))
    assert_array_equal(labels, expected)

    # Test wrong input arrays
    wrong_arr = rng.random((5, 7))
    with pytest.raises(ValueError):
        tmp.fit(wrong_arr)
