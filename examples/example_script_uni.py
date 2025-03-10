"""Example script for running onion_uni."""

# Author: Becchi Matteo <bechmath@gmail.com>

import matplotlib.pyplot as plt
import numpy as np

from tropea_clustering import helpers, onion_uni
from tropea_clustering.plot import (
    plot_medoids_uni,
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
PATH_TO_INPUT_DATA = "onion_example_files/data/univariate_time-series.npy"

### Load the input data - it's an array of shape (n_particles, n_frames)
input_data = np.load(PATH_TO_INPUT_DATA)[:, 1:]
n_particles, n_frames = input_data.shape

### CLUSTERING WITH A SINGLE TIME RESOLUTION ###
### Chose the time resolution --> the length of the windows in which the
### time-series will be divided
TAU_WINDOW = 5
n_windows = int(n_frames / TAU_WINDOW)  # Number of windows

### The input array needs to be (n_particles * n_windows, TAU_WINDOW) because
### each window is treated as a single data-point
reshaped_data = helpers.reshape_from_nt(input_data, TAU_WINDOW)

### onion_uni() returns the list of states and the label for each
### signal window
state_list, labels = onion_uni(reshaped_data)

### These functions are examples of how to visualize the results
plot_output_uni("output_uni/Fig1.png", reshaped_data, n_particles, state_list)
plot_one_trj_uni(
    "output_uni/Fig2.png", 1234, reshaped_data, n_particles, labels
)
plot_medoids_uni("output_uni/Fig3.png", reshaped_data, labels)
plot_state_populations("output_uni/Fig4.png", n_particles, TAU_WINDOW, labels)
plot_sankey("output_uni/Fig5.png", labels, n_particles, [10, 20, 30, 40])

### CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS ###
TAU_WINDOWS = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))

tra = np.zeros((len(TAU_WINDOWS), 3))  # List of number of states and
# ENV0 population for each tau_window
list_of_pop = []  # List of the states' population for each tau_window

for i, tau_window in enumerate(TAU_WINDOWS):
    reshaped_data = helpers.reshape_from_nt(input_data, tau_window)

    state_list, labels = onion_uni(reshaped_data)

    pop_list = [state.perc for state in state_list]
    pop_list.insert(0, 1 - np.sum(np.array(pop_list)))
    list_of_pop.append(pop_list)

    tra[i][0] = tau_window
    tra[i][1] = len(state_list)
    tra[i][2] = pop_list[0]

### These functions are examples of how to visualize the results
plot_time_res_analysis("output_uni/Fig6.png", tra)
plot_pop_fractions("output_uni/Fig7.png", list_of_pop, tra)

plt.show()
