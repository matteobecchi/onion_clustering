"""Pytest for onion_uni and OnionUni."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from tropea_clustering import OnionUni, helpers, onion_uni


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
    random_walk = []
    for _ in range(N_PARTICLES):
        tmp = [0.0]
        for _ in range(N_STEPS - 1):
            d_x = rng.normal()
            x_new = tmp[-1] + d_x
            tmp.append(x_new)
        random_walk.append(tmp)

    reshaped_input_data = helpers.reshape_from_nt(
        np.array(random_walk), TAU_WINDOW
    )

    with tempfile.TemporaryDirectory() as _:
        # Test the class methods
        tmp = OnionUni()
        tmp_params = {"bins": 100, "number_of_sigmas": 2.0}
        tmp.set_params(**tmp_params)
        _ = tmp.get_params()
        tmp.fit_predict(reshaped_input_data)

        state_list, labels = onion_uni(reshaped_input_data)

        _ = state_list[0].get_attributes()

        # Define the paths to the expected output files
        original_dir = original_wd / "test/"
        expected_output_path = original_dir / "output_uni/labels.npy"

        # np.save(expected_output_path, labels)

        # Compare the contents of the expected and actual output
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels)
