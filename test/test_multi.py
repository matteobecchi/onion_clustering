"""Pytest for onion_multi and OnionMulti."""

import os
import tempfile

import numpy as np
import pytest

from tropea_clustering import OnionMulti, onion_multi


# Define a fixture to set up the test environment
@pytest.fixture
def setup_test_environment(tmpdir):
    # tmpdir is a built-in pytest fixture providing a temporary directory
    original_dir = os.getcwd()  # Save the current working directory
    os.chdir(str(tmpdir))  # Change to the temporary directory
    yield tmpdir
    os.chdir(
        original_dir
    )  # Restore the original working directory after the test


# Define the actual test
def test_output_files(setup_test_environment):
    ### Set all the analysis parameters ###
    N_PARTICLES = 5
    N_STEPS = 1000
    TAU_WINDOW = 10

    ## Create the input data ###
    rng = np.random.default_rng(12345)
    random_walk_x = []
    random_walk_y = []
    for _ in range(N_PARTICLES):
        tmp_x, tmp_y = [0.0], [0.0]
        for _ in range(N_STEPS - 1):
            d_x = rng.normal()
            x_new = tmp_x[-1] + d_x
            tmp_x.append(x_new)
            d_y = rng.normal()
            y_new = tmp_y[-1] + d_y
            tmp_y.append(y_new)
        random_walk_x.append(tmp_x)
        random_walk_y.append(tmp_y)

    n_windows = int(N_STEPS / TAU_WINDOW)

    reshaped_input_data_x = np.reshape(
        np.array(random_walk_x), (N_PARTICLES * n_windows, -1)
    )
    reshaped_input_data_y = np.reshape(
        np.array(random_walk_y), (N_PARTICLES * n_windows, -1)
    )
    reshaped_input_data = np.array(
        [
            np.concatenate((tmp, reshaped_input_data_y[i]))
            for i, tmp in enumerate(reshaped_input_data_x)
        ]
    )

    with tempfile.TemporaryDirectory() as _:
        # Call your code to generate the output files
        tmp = OnionMulti()
        tmp.fit_predict(reshaped_input_data)
        _ = tmp.get_params()
        tmp.set_params()

        state_list, labels = onion_multi(reshaped_input_data)

        _ = state_list[0].get_attributes()

        # Define the paths to the expected output files
        original_dir = "/Users/mattebecchi/onion_clustering/test/"
        expected_output_path = original_dir + "output_multi/labels.npy"

        np.save(expected_output_path, labels)

        # Compare the contents of the expected and actual output
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels)
