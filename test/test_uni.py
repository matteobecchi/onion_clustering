"""Pytest for onion_uni and OnionUni."""

import os

import numpy as np
from numpy.testing import assert_array_equal

from tropea_clustering import OnionUni, helpers, onion_uni


def test_onion_uni():
    n_particles = 5
    n_steps = 1000
    delta_t = 10

    rng = np.random.default_rng(12345)
    random_walk = []
    for _ in range(n_particles):
        tmp = [0.0]
        for _ in range(n_steps - 1):
            d_x = rng.normal()
            x_new = tmp[-1] + d_x
            tmp.append(x_new)
        random_walk.append(tmp)

    reshaped_input_data = helpers.reshape_from_nt(
        np.array(random_walk), delta_t
    )

    # Test class interface
    tmp = OnionUni()
    tmp_params = {"bins": 100, "number_of_sigmas": 2.0}
    tmp.set_params(**tmp_params)
    _ = tmp.get_params()
    tmp.fit_predict(reshaped_input_data)

    # Test functional interface
    state_list, labels = onion_uni(reshaped_input_data)
    _ = state_list[0].get_attributes()

    # Check clustering output
    this_dir = os.path.dirname(__file__)
    expected = np.load(os.path.join(this_dir, "output_uni", "labels.npy"))
    assert_array_equal(labels, expected)
