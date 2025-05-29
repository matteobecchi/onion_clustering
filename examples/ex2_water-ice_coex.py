"""Analysis of LENS trajectories on water/ice coexistence."""

# Author: Becchi Matteo <bechmath@gmail.com>

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tropea_clustering import OnionUniSmooth
from tropea_clustering.plot import (
    plot_one_trj_uni,
    plot_output_uni,
    plot_pop_fractions,
    plot_sankey,
    plot_state_populations,
    plot_time_res_analysis,
)


def main():
    # Use git clone git@github.com:matteobecchi/onion_example_files.git
    # to download example datasets
    path_to_input = "onion_example_files/data/univariate_time-series.npy"
    output_path = Path("ex2_water-ice_coex")

    # Load the input data - it's an array of shape (n_particles, n_frames)
    lens = np.load(path_to_input)[:, 1:]
    n_particles, n_frames = lens.shape

    # Apply OnionUni on a wide range of time resolutions
    delta_t_list = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))
    results = np.zeros((delta_t_list.size, 3))
    list_of_pops = []

    for i, delta_t in enumerate(delta_t_list):
        on_cl = OnionUniSmooth(delta_t)
        on_cl.fit(lens)
        state_list, labels = on_cl.state_list_, on_cl.labels_

        results[i][0] = delta_t
        results[i][1] = len(state_list)
        results[i][2] = np.sum(labels == -1) / labels.size
        pops = [np.sum(labels == j) / labels.size for j in np.unique(labels)]
        list_of_pops.append(pops)

    plot_time_res_analysis(output_path / "Fig1.png", results)
    plot_pop_fractions(output_path / "Fig2.png", list_of_pops, results)

    # Perform clustering at delta_t = 5 frames
    on_cl = OnionUniSmooth(delta_t=5)
    on_cl.fit(lens)
    state_list, labels = on_cl.state_list_, on_cl.labels_

    plot_output_uni(output_path / "Fig3.png", lens, state_list)
    plot_one_trj_uni(output_path / "Fig4.png", 1234, lens, labels)
    plot_state_populations(output_path / "Fig5.png", labels)
    plot_sankey(output_path / "Fig6.png", labels, [10, 20, 30, 40])

    plt.show()


if __name__ == "__main__":
    main()
