"""Pytest for onion_multi_smooth and OnionMultiSmooth."""

from pathlib import Path

import numpy as np

from tropea_clustering import onion_multi_smooth, plot_smooth


def main():
    np.random.seed(42)
    mu1 = np.array([0.0, 0.0])
    cov1 = np.array([[0.05, 0.02], [0.02, 0.02]])
    mu2 = np.array([1.0, 1.0])
    cov2 = np.array([[0.1, -0.03], [-0.03, 0.05]])
    list_data_2d = []
    for _ in range(100):
        points1 = np.random.multivariate_normal(mu1, cov1, size=500)
        points2 = np.random.multivariate_normal(mu2, cov2, size=500)
        time_series = np.vstack([points1, points2])
        list_data_2d.append(time_series)
    input_data_2d = np.array(list_data_2d)

    delta_t = 25
    state_list, labels = onion_multi_smooth(input_data_2d, delta_t)

    plot_smooth.plot_output_multi(
        Path("tmp1.png"), input_data_2d, state_list, labels
    )
    # plot_smooth.plot_one_trj_multi(Path("tmp2.png"), 0, input_data_2d, labels)


if __name__ == "__main__":
    main()
