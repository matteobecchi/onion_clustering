"""Example script for running onion_uni."""

# Author: Becchi Matteo <bechmath@gmail.com>

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tropea_clustering import onion_uni
from tropea_clustering.plot import (
    plot_one_trj_uni,
    plot_output_uni,
    plot_pop_fractions,
    plot_sankey,
    plot_state_populations,
    plot_time_res_analysis,
)

#############################################################################
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
path_to_input = "onion_example_files/data/univariate_time-series.npy"

### Load the input data - it's an array of shape (n_particles, n_frames)
input_data = np.load(path_to_input)[:, 1:]
n_particles, n_frames = input_data.shape

### CLUSTERING WITH A SINGLE TIME RESOLUTION ###
### Chose the time resolution --> the length of the windows in which the
### time-series will be divided
delta_t = 5

### onion_uni() returns the list of states and the label for each
### signal window
state_list, labels = onion_uni(input_data, delta_t)

### These functions are examples of how to visualize the results
output_path = Path("output_uni")
plot_output_uni(output_path / "Fig1.png", input_data, state_list)
plot_one_trj_uni(output_path / "Fig2.png", 1234, input_data, labels)
# plot_medoids_uni(output_path / "Fig3.png", input_data, labels)
plot_state_populations(output_path / "Fig4.png", labels)
plot_sankey(output_path / "Fig5.png", labels, [10, 20, 30, 40])

### CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS ###
delta_t_list = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))

tra = np.zeros((delta_t_list.size, 3))  # List of number of states and
# ENV0 population for each tau_window
list_of_pop = []  # List of the states' population for each tau_window

for i, delta_t in enumerate(delta_t_list):
    state_list, labels = onion_uni(input_data, delta_t)

    pop_list = [state.perc for state in state_list]
    pop_list.insert(0, 1 - np.sum(np.array(pop_list)))
    list_of_pop.append(pop_list)

    tra[i][0] = delta_t
    tra[i][1] = len(state_list)
    tra[i][2] = pop_list[0]

### These functions are examples of how to visualize the results
plot_time_res_analysis(output_path / "Fig6.png", tra)
plot_pop_fractions(output_path / "Fig7.png", list_of_pop, tra)

plt.show()
