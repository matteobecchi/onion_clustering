"""Example script for running onion_multi."""

# Author: Becchi Matteo <bechmath@gmail.com>

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tropea_clustering import onion_multi
from tropea_clustering.plot import (
    plot_medoids_multi,
    plot_one_trj_multi,
    plot_output_multi,
    plot_pop_fractions,
    plot_sankey,
    plot_state_populations,
    plot_time_res_analysis,
)

##############################################################################
# Use git clone git@github.com:matteobecchi/onion_example_files.git
# to download example datasets
path_to_input = "onion_example_files/data/multivariate_time-series.npy"

### Load the input data - it's an array of shape
### (n_dims, n_particles, n_frames)
input_data = np.load(path_to_input)
n_dims, n_particles, n_frames = input_data.shape

### CLUSTERING WITH A SINGLE TIME RESOLUTION ###
### Chose the time resolution --> the length of the windows in which the
### time-series will be divided
delta_t = 10
bins = 25  # For mutlivariate clustering, setting BINS is often important

### onion_multi() returns the list of states and the label for each
### signal window
state_list, labels = onion_multi(input_data, delta_t, bins=bins)

### These functions are examples of how to visualize the results
output_path = Path("output_multi")
plot_output_multi(output_path / "Fig1.png", input_data, state_list, labels)
plot_one_trj_multi(output_path / "Fig2.png", 0, input_data, labels)
plot_medoids_multi(output_path / "Fig3.png", input_data, labels)
plot_state_populations(output_path / "Fig4.png", labels)
plot_sankey(output_path / "Fig5.png", labels, [100, 200, 300, 400])

### CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS ###
delta_t_list = np.geomspace(3, n_frames, 20, dtype=int)

tra = np.zeros((delta_t_list.size, 3))  # List of number of states and
# ENV0 population for each tau_window
pop_list = []  # List of the states' population for each tau_window

for i, tau_window in enumerate(delta_t_list):
    state_list, labels = onion_multi(input_data, bins=bins)

    list_pop = [state.perc for state in state_list]
    list_pop.insert(0, 1 - np.sum(np.array(list_pop)))

    tra[i][0] = delta_t
    tra[i][1] = len(state_list)
    tra[i][2] = list_pop[0]
    pop_list.append(list_pop)

### These functions are examples of how to visualize the results
plot_time_res_analysis(output_path / "Fig6.png", tra)
plot_pop_fractions(output_path / "Fig7.png", pop_list, tra)

plt.show()
