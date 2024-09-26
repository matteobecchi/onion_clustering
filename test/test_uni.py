"""Pytest for onion_uni and OnionUni."""

import os
import tempfile

import numpy as np
import pytest

from onion_clustering.onion_uni import OnionUni, onion_uni


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
    random_walk = []
    for _ in range(N_PARTICLES):
        tmp = [0.0]
        for _ in range(N_STEPS - 1):
            d_x = rng.normal()
            x_new = tmp[-1] + d_x
            tmp.append(x_new)
        random_walk.append(tmp)

    n_windows = int(N_STEPS / TAU_WINDOW)
    reshaped_input_data = np.reshape(
        np.array(random_walk), (N_PARTICLES * n_windows, -1)
    )

    wrong_arr_1 = rng.normal((3, 3, 3))
    wrong_arr_2 = rng.normal((1, 10))
    wrong_arr_3 = rng.normal((3, 3)) + 1.0j * rng.normal((3, 3))

    with tempfile.TemporaryDirectory() as _:
        # Call your code to generate the output files
        tmp = OnionUni()
        tmp.fit_predict(reshaped_input_data)
        _ = tmp.get_params()
        tmp.set_params()

        # Test wrong input arrays
        with pytest.raises(ValueError):
            _, _ = onion_uni(wrong_arr_1)
            _, _ = onion_uni(wrong_arr_2)
            _, _ = onion_uni(wrong_arr_2.T)
            _, _ = onion_uni(wrong_arr_3)

        _, labels = onion_uni(reshaped_input_data)

        # Define the paths to the expected output files
        original_dir = "/Users/mattebecchi/onion_clustering/test/"
        expected_output_path = original_dir + "output_uni/labels.npy"

        # np.save(expected_output_path, labels)

        # Compare the contents of the expected and actual output
        expected_output = np.load(expected_output_path)
        print(np.sum(expected_output != labels) / labels.size)
        assert np.allclose(expected_output, labels)
