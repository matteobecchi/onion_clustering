"""
Example script for running onion_multi.
"""

import matplotlib.pyplot as plt
import numpy as np

from tropea_clustering import helpers, onion_multi
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
PATH_TO_INPUT_DATA = "onion_example_files/data/multivariate_time-series.npy"

### Load the input data - it's an array of shape
### (n_dims, n_particles, n_frames)
input_data = np.load(PATH_TO_INPUT_DATA)
n_dims = input_data.shape[0]
n_particles = input_data.shape[1]
n_frames = input_data.shape[2]

### CLUSTERING WITH A SINGLE TIME RESOLUTION ###
### Chose the time resolution --> the length of the windows in which the
### time-series will be divided
TAU_WINDOW = 10
BINS = 25  # For mutlivariate clustering, setting BINS is often important
n_windows = int(n_frames / TAU_WINDOW)  # Number of windows

### The input array has to be (n_parrticles * n_windows, TAU_WINDOW * n_dims)
### because each window is trerated as a single data-point
reshaped_data = helpers.reshape_from_dnt(input_data, TAU_WINDOW)

### onion_multi() returns the list of states and the label for each
### signal window
state_list, labels = onion_multi(reshaped_data, bins=BINS)

### These functions are examples of how to visualize the results
plot_output_multi(
    "output_multi/Fig1.png", input_data, state_list, labels, TAU_WINDOW
)
plot_one_trj_multi("output_multi/Fig2.png", 0, TAU_WINDOW, input_data, labels)
plot_medoids_multi("output_multi/Fig3.png", TAU_WINDOW, input_data, labels)
plot_state_populations("output_multi/Fig4.png", n_windows, labels)
plot_sankey("output_multi/Fig5.png", labels, n_windows, [100, 200, 300, 400])

### CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS ###
TAU_WINDOWS_LIST = np.geomspace(3, 10000, 20, dtype=int)

tra = np.zeros((len(TAU_WINDOWS_LIST), 3))  # List of number of states and
# ENV0 population for each tau_window
pop_list = []  # List of the states' population for each tau_window

for i, tau_window in enumerate(TAU_WINDOWS_LIST):
    reshaped_data = helpers.reshape_from_dnt(input_data, tau_window)

    state_list, labels = onion_multi(reshaped_data, bins=BINS)

    list_pop = [state.perc for state in state_list]
    list_pop.insert(0, 1 - np.sum(np.array(list_pop)))

    tra[i][0] = tau_window
    tra[i][1] = len(state_list)
    tra[i][2] = list_pop[0]
    pop_list.append(list_pop)

### These functions are examples of how to visualize the results
plot_time_res_analysis("output_multi/Fig6.png", tra)
plot_pop_fractions("output_multi/Fig7.png", pop_list)

plt.show()
