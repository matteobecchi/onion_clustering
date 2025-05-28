"""Pytest for onion_uni and OnionUni."""

# Author: Matteo Becchi <bechmath@gmail.com>

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from tropea_clustering import OnionUni, onion_uni


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
    rng = np.random.default_rng(12345)
    input_data = np.array(
        [
            np.concatenate(
                (rng.normal(0.0, 0.1, 500), rng.normal(1.0, 0.1, 500))
            )
            for _ in range(100)
        ]
    )

    delta_t = 10

    with tempfile.TemporaryDirectory() as _:
        # Test the class methods
        on_cl = OnionUni(delta_t)
        tmp_params = {"bins": "auto", "number_of_sigmas": 3.0}
        on_cl.set_params(**tmp_params)
        _ = on_cl.get_params()
        on_cl.fit(input_data)
        on_cl.fit_predict(input_data)

        # Test the function
        state_list, labels = onion_uni(input_data, delta_t)

        _ = state_list[0].get_attributes()

        # Define the paths to the expected output files
        original_dir = original_wd / "test/"
        expected_output_path = original_dir / "output_uni/labels.npy"

        # np.save(expected_output_path, labels)

        # Compare the contents of the expected and actual output
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels)

        # Test wrong input
        input_data = np.ones(10)  # 1D array
        with pytest.raises(ValueError):
            on_cl.fit(input_data)

        input_data = np.zeros((100, 1))  # just one frame
        with pytest.raises(ValueError):
            on_cl.fit(input_data)
