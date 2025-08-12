"""Explore the effect of the parameters of OnionMulti."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tropea_clustering import OnionMultiSmooth
from tropea_clustering.plot_smooth import (
    plot_pop_fractions,
    plot_time_res_analysis,
)


def gaussian_2d(x, y, mean, cov):
    """Evaluate a 2D Gaussian (unnormalized) at point (x, y)."""
    pos = np.array([x, y])
    diff = pos - mean
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * diff @ inv_cov @ diff
    return np.exp(exponent)


def energy_landscape(x: float, y: float) -> float:
    """A 2-dimensional potential energy landscape with 1 minima."""
    mean = np.array([0.0, 0.0])
    cov = np.array([[0.1, 0.0], [0.00, 0.1]])
    gauss = gaussian_2d(x, y, mean, cov)
    return -np.log(gauss + 1e-6)


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
    particles = rng.standard_normal((n_atoms, 2)) * 0.1

    trajectory = np.zeros((time_steps, n_atoms, 2))
    for t in tqdm(range(time_steps)):
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


def effect_of_delta_t(
    dataset: NDArray[np.float64],
    data_path: Path,
) -> None:
    """Explore the effect of the time resolution ∆t."""
    delta_t_list = np.unique(np.geomspace(2, 2000, 40, dtype=int))
    tra = np.zeros((len(delta_t_list), 3))
    list_of_pop = []
    for i, delta_t in enumerate(delta_t_list):
        on_cl = OnionMultiSmooth(delta_t=delta_t).fit(dataset)
        labels = on_cl.labels_
        tra[i][0] = delta_t
        tra[i][1] = len(on_cl.state_list_)
        tra[i][2] = np.sum(labels == -1) / labels.size
        pops = [np.sum(labels == j) / labels.size for j in np.unique(labels)]
        list_of_pop.append(pops)

    # We can fit the relation unclassified_fraction - ∆t:
    def sigmoidal(
        t: NDArray[np.float64],
        t0: float,
        tau: float,
    ) -> NDArray[np.float64]:
        return 1 / (1 + np.exp(-(t - t0) / tau))

    plot_time_res_analysis(title=data_path / "delta_t_1.png", tra=tra)
    plot_pop_fractions(
        title=data_path / "delta_t_2.png",
        list_of_pop=list_of_pop,
        tra=tra,
    )


def effect_of_sigma_threshold(
    dataset: NDArray[np.float64],
    data_path: Path,
) -> None:
    """Explore the effect of the number_of_sigma parameter."""
    sigma_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    tra = np.zeros((len(sigma_list), 3))
    list_of_pop = []
    for i, nos in enumerate(sigma_list):
        on_cl = OnionMultiSmooth(delta_t=80, number_of_sigmas=nos).fit(dataset)
        labels = on_cl.labels_
        tra[i][0] = nos
        tra[i][1] = len(on_cl.state_list_)
        tra[i][2] = np.sum(labels == -1) / labels.size
        pops = [np.sum(labels == j) / labels.size for j in np.unique(labels)]
        list_of_pop.append(pops)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(sigma_list, tra.T[1], marker="o")
    ax[0].set_ylabel("Number of states")
    ax[1].plot(sigma_list, tra.T[2], marker="o", c="C1")
    ax[1].set_ylabel("Unclassified fraction")
    ax[1].set_xlabel("Number of sigmas")
    fig.savefig(data_path / "nos.png", dpi=600)
    plt.close()


def main():
    """Explore the effect of the parameters of OnionMulti."""
    data_path = Path("ex3_params_effect")

    # Load or create the input dataset
    n_atoms = 100
    n_frames = 10000
    file_path = data_path / "ex3_data1.npy"
    if file_path.exists():
        dataset = np.load(file_path)
    else:
        dataset = create_trajectory(n_atoms, n_frames, file_path)

    # effect_of_delta_t(dataset, data_path)
    effect_of_sigma_threshold(dataset, data_path)


if __name__ == "__main__":
    main()
