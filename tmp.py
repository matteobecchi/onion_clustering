import numpy as np

from tropea_clustering import onion_uni

# Select time resolution
delta_t = 5

# Create random input data
np.random.seed(1234)
n_particles = 5
n_steps = 1000

input_data = np.random.rand(n_particles, n_steps)

# Run Onion Clustering
state_list, labels = onion_uni(input_data, delta_t)

print(state_list[0].mean)
