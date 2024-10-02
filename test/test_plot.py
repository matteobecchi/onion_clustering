"""Pytest for plot."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from tropea_clustering import onion_multi, onion_uni, plot


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
    N_PARTICLES = 2
    N_STEPS = 500
    TAU_WINDOW = 10

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

    reshaped_data_uni = np.reshape(
        np.array(random_walk_x), (N_PARTICLES * n_windows, -1)
    )

    reshaped_input_data_x = np.reshape(
        np.array(random_walk_x), (N_PARTICLES * n_windows, -1)
    )
    reshaped_input_data_y = np.reshape(
        np.array(random_walk_y), (N_PARTICLES * n_windows, -1)
    )
    reshaped_data_multi = np.array(
        [
            np.concatenate((tmp, reshaped_input_data_y[i]))
            for i, tmp in enumerate(reshaped_input_data_x)
        ]
    )

    with tempfile.TemporaryDirectory() as _:
        state_list, labels = onion_uni(reshaped_data_uni)

        plot.plot_output_uni(
            "tmp_fig.png", reshaped_data_uni, n_windows, state_list
        )
        plot.plot_one_trj_uni(
            "tmp_fig.png", 0, reshaped_data_uni, labels, n_windows
        )
        plot.plot_medoids_uni("tmp_fig.png", reshaped_data_uni, labels)
        plot.plot_state_populations("tmp_fig.png", n_windows, labels)
        plot.plot_sankey("tmp_fig.png", labels, n_windows, [1, 3, 5, 7])

        state_list, labels = onion_multi(reshaped_data_multi)

        old_input_data = np.array([random_walk_x, random_walk_y])
        plot.plot_output_multi(
            "tmp_fig.png", old_input_data, state_list, labels, TAU_WINDOW
        )
        plot.plot_one_trj_multi(
            "tmp_fig.png", 0, TAU_WINDOW, old_input_data, labels
        )
        plot.plot_medoids_multi(
            "tmp_fig.png", TAU_WINDOW, old_input_data, labels
        )
