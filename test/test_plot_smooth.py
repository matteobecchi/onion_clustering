"""Pytest for plot."""

# Author: Matteo Becchi <bechmath@gmail.com>

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from tropea_clustering import onion_multi_smooth, onion_uni_smooth, plot_smooth


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
    rng = np.random.default_rng(12345)
    input_data = np.array(
        [
            np.concatenate(
                (rng.normal(0.0, 0.1, 500), rng.normal(1.0, 0.1, 500))
            )
            for _ in range(100)
        ]
    )
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

    delta_t = 10

    with tempfile.TemporaryDirectory() as _:
        out_path = Path("tmp_fig.png")

        state_list_u, labels = onion_uni_smooth(input_data, delta_t)

        plot_smooth.plot_output_uni(out_path, input_data, state_list_u)
        plot_smooth.plot_one_trj_uni(out_path, 0, input_data, labels)
        plot_smooth.plot_state_populations(out_path, labels)
        plot_smooth.plot_sankey(out_path, labels, [1, 3, 5, 7])

        state_list_m, labels = onion_multi_smooth(input_data_2d, delta_t)

        plot_smooth.plot_output_multi(
            out_path, input_data_2d, state_list_m, labels
        )
        plot_smooth.plot_one_trj_multi(out_path, 0, input_data_2d, labels)
