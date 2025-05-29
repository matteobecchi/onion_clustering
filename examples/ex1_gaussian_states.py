"""Analysis of particles moving between Gaussian energy minima.

In this example, a dataset is created composed by the (x, y) coordinates
of particles moving in a 2D energy landscape with 4 energy minima.

Onion clustering is applied initially to the univariate dataset of x
positions, finding only two clusters.

Then, applying is on the full (x, y) bivariate dataset, all the 4 minima are
found.
"""

# Author: Becchi Matteo <bechmath@gmail.com>

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tropea_clustering import OnionMultiSmooth, OnionUniSmooth
from tropea_clustering.plot import (
    plot_one_trj_multi,
    plot_one_trj_uni,
    plot_output_multi,
    plot_output_uni,
    plot_pop_fractions,
    plot_state_populations,
    plot_time_res_analysis,
)


def energy_landscape(x: float, y: float) -> float:
    """A 2-dimensional potential energy landscape with 4 minima."""
    sigma = 0.12  # Width of the Gaussian wells
    gauss1 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gauss2 = np.exp(-((x - 1) ** 2 + y**2) / (2 * sigma**2))
    gauss3 = np.exp(-(x**2 + (y - 1) ** 2) / (2 * sigma**2))
    gauss4 = np.exp(-((x - 1) ** 2 + (y - 1) ** 2) / (2 * sigma**2))
    return -np.log(gauss1 + gauss2 + gauss3 + gauss4 + 1e-6)


def numerical_gradient(
    x: float, y: float, h: float = 1e-5
) -> tuple[float, float]:
    """Compute numerical gradient using finite differences."""
    grad_x = (energy_landscape(x + h, y) - energy_landscape(x - h, y)) / (
        2 * h
    )
    grad_y = (energy_landscape(x, y + h) - energy_landscape(x, y - h)) / (
        2 * h
    )
    return -grad_x, -grad_y


def create_trajectory(
    n_atoms: int,
    time_steps: int,
    file_path: Path,
) -> NDArray[np.float64]:
    """Simulate Langevin Dynamics on a given energy landscape."""
    rng = np.random.default_rng(0)
    dt = 0.01  # Time step
    diffusion_coeff = 0.8  # Diffusion coefficient (random noise strength)

    # Initialize particles' positions
    particles = rng.standard_normal((n_atoms, 2)) * 0.2
    n_group = n_atoms // 4
    particles[n_group : 2 * n_group, 1] += 1  # (0, 1)
    particles[2 * n_group : 3 * n_group, 0] += 1  # (1, 0)
    particles[3 * n_group :, 0] += 1  # (1, 1)
    particles[3 * n_group :, 1] += 1

    trajectory = np.zeros((time_steps, n_atoms, 2))
    for t in range(time_steps):
        for i in range(n_atoms):
            x, y = particles[i]
            fx, fy = numerical_gradient(x, y)
            noise_x = np.sqrt(2 * diffusion_coeff * dt) * rng.standard_normal()
            noise_y = np.sqrt(2 * diffusion_coeff * dt) * rng.standard_normal()

            # Update position with deterministic force and stochastic term
            particles[i, 0] += fx * dt + noise_x
            particles[i, 1] += fy * dt + noise_y

            trajectory[t, i] = particles[i]

    plt.figure()
    plt.plot(trajectory[:, :, 0], trajectory[:, :, 1])
    plt.show()

    dataset = np.transpose(trajectory, (1, 0, 2))
    np.save(file_path, dataset)
    return dataset


def main():
    """Analysis of particles moving between Gaussian energy minima."""
    data_path = Path("ex1_gaussian_states")

    # Load or create the input dataset
    n_atoms = 100
    n_frames = 10000
    file_path = data_path / "ex1_data1.npy"
    if file_path.exists():
        dataset = np.load(file_path)
    else:
        dataset = create_trajectory(n_atoms, n_frames, file_path)

    delta_t_list = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))

    # Test OnionUni on a wide range of time resolutions
    data_1d = dataset[:, :, 0]
    results = np.zeros((delta_t_list.size, 3))
    list_of_pops = []

    for i, delta_t in enumerate(delta_t_list):
        on_cl = OnionUniSmooth(delta_t)
        on_cl.fit(data_1d)
        state_list, labels = on_cl.state_list_, on_cl.labels_

        results[i][0] = delta_t
        results[i][1] = len(state_list)
        results[i][2] = np.sum(labels == -1) / labels.size
        pops = [np.sum(labels == j) / labels.size for j in np.unique(labels)]
        list_of_pops.append(pops)

    plot_time_res_analysis(data_path / "Fig1.png", results)
    plot_pop_fractions(data_path / "Fig2.png", list_of_pops, results)

    # Perform clustering at delta_t = 100 frames
    on_cl = OnionUniSmooth(delta_t=100)
    on_cl.fit(data_1d)
    state_list, labels = on_cl.state_list_, on_cl.labels_

    plot_output_uni(data_path / "Fig3.png", data_1d, state_list)
    plot_one_trj_uni(data_path / "Fig4.png", 10, data_1d, labels)
    plot_state_populations(data_path / "Fig5.png", labels)

    # Test OnionMulti on a wide range of time resolutions
    results = np.zeros((delta_t_list.size, 3))
    list_of_pops = []

    for i, delta_t in enumerate(delta_t_list):
        on_cl = OnionMultiSmooth(delta_t)
        on_cl.fit(dataset)
        state_list, labels = on_cl.state_list_, on_cl.labels_

        results[i][0] = delta_t
        results[i][1] = len(state_list)
        results[i][2] = np.sum(labels == -1) / labels.size
        pops = [np.sum(labels == j) / labels.size for j in np.unique(labels)]
        list_of_pops.append(pops)

    plot_time_res_analysis(data_path / "Fig6.png", results)
    plot_pop_fractions(data_path / "Fig7.png", list_of_pops, results)

    # Perform clustering at delta_t = 100 frames
    on_cl = OnionMultiSmooth(delta_t=10)
    on_cl.fit(dataset)
    state_list, labels = on_cl.state_list_, on_cl.labels_

    plot_output_multi(data_path / "Fig8.png", dataset, state_list, labels)
    plot_one_trj_multi(data_path / "Fig9.png", 10, dataset, labels)
    plot_state_populations(data_path / "Fig10.png", labels)

    plt.show()


if __name__ == "__main__":
    main()
