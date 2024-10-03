"""Pytest for onion_multi and OnionMulti."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from tropea_clustering import OnionMulti, helpers, onion_multi


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()
    # Ensure the original working directory is restored after the test
    yield original_dir
    os.chdir(original_dir)


@pytest.fixture
def temp_dir(original_wd: Path) -> Generator[Path, None, None]:
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Change the working directory to the temporary one
        os.chdir(tmp_path)
        # Yield the temporary directory path for use in the test
        yield tmp_path
        # The context manager ensures that the directory is cleaned up
        os.chdir(original_wd)  # Restore the original working directory


# Define the actual test
def test_output_files(original_wd: Path, temp_dir: Path):
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
    input_data = np.array([random_walk_x, random_walk_y])

    reshaped_input_data = helpers.reshape_from_dnt(input_data, TAU_WINDOW)

    wrong_arr = rng.random((5, 7))

    with tempfile.TemporaryDirectory() as _:
        # Test the class methods
        tmp = OnionMulti()
        tmp_params = {"bins": 50, "number_of_sigmas": 2.0}
        tmp.set_params(**tmp_params)
        _ = tmp.get_params()
        tmp.fit_predict(reshaped_input_data)

        # Test wrong input arrays
        with pytest.raises(ValueError):
            tmp.fit(wrong_arr)

        state_list, labels = onion_multi(reshaped_input_data)

        _ = state_list[0].get_attributes()

        # Define the paths to the expected output files
        original_dir = original_wd / "test/"
        expected_output_path = original_dir / "output_multi/labels.npy"

        np.save(expected_output_path, labels)

        # Compare the contents of the expected and actual output
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels)
