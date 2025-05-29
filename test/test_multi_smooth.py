"""Pytest for onion_multi and OnionMulti."""

# Author: Matteo Becchi <bechmath@gmail.com>

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from tropea_clustering import OnionMultiSmooth, onion_multi_smooth


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
    wrong_arr = rng.random((5, 7))

    with tempfile.TemporaryDirectory() as _:
        # Test the class methods
        on_cl = OnionMultiSmooth(delta_t)
        tmp_params = {"bins": 50, "number_of_sigmas": 2.0}
        on_cl.set_params(**tmp_params)
        _ = on_cl.get_params()
        _ = on_cl.fit_predict(input_data_2d)

        # Test wrong input arrays
        with pytest.raises(ValueError):
            on_cl.fit(wrong_arr)

        state_list, labels = onion_multi_smooth(input_data_2d, delta_t)

        _ = state_list[0].get_attributes()

        # Define the paths to the expected output files
        original_dir = original_wd / "test/"
        expected_output_path = original_dir / "output_multi_smooth/labels.npy"

        # Compare the contents of the expected and actual output
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels)

        # # Test if also the 3D case works
        # state_list, labels = onion_multi(input_data_3d, delta_t)
        # expected_output_path = original_dir / "output_multi/labels_3D.npy"
        # expected_output = np.load(expected_output_path)
        # assert np.allclose(expected_output, labels)

        # Test wrong input
        input_data = np.ones((10, 10))  # 2D array
        with pytest.raises(
            ValueError, match="Expected 3-dimensional input data."
        ):
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
