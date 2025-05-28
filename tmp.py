import numpy as np

from tropea_clustering import OnionMulti

# Select time resolution
delta_t = 2

# Create random input data
np.random.seed(1234)
n_features = 2
n_particles = 5
n_steps = 1000

input_data = np.random.rand(n_particles, n_steps, n_features)

# Run Onion Clustering
clusterer = OnionMulti(delta_t)
clust_params = {"bins": 100, "number_of_sigmas": 2.0}
clusterer.set_params(**clust_params)
clusterer.fit(input_data)

print(clusterer.state_list_[0].mean[0])
