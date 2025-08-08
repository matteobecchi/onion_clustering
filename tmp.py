"""Pytest for onion_multi_smooth and OnionMultiSmooth."""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tropea_clustering import OnionMultiSmooth, onion_multi_smooth, plot_smooth


def main():
    rng = np.random.default_rng(12345)
    input_data_2d = np.array(
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

    delta_t = 5

    # Test functional interface
    state_list, labels = onion_multi_smooth(input_data_2d, delta_t)

    # plot_smooth.plot_output_multi(
    #     Path("tmp1.png"), input_data_2d, state_list, labels
    # )
    # plot_smooth.plot_one_trj_multi(Path("tmp2.png"), 0, input_data_2d, labels)


if __name__ == "__main__":
    main()
