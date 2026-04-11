"""Pytest for Onion class."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tropea_clustering import OnionClustering, onion_clustering


def _plot_cluster_labels(
    X: np.ndarray,
    labels: np.ndarray,
    frame: int | None = 0,
    title: str | None = None,
) -> None:
    """Plot 2D points colored by cluster label.

    If frame is None, the first two axes of X are flattened and all points are
    plotted together. Otherwise, a single time frame is plotted.
    """

    if X.ndim != 3 or X.shape[2] != 2:
        raise ValueError(
            "X must be a 3D array with the last dimension equal to 2."
        )
    if labels.shape != X.shape[:2] and labels.shape != (np.prod(X.shape[:2]),):
        raise ValueError(
            "labels must have shape (n1, n2) or (n1*n2,) and match X."
        )

    if frame is None:
        X_flat = X.reshape(-1, 2)
        labels_flat = labels.reshape(-1) if labels.ndim == 2 else labels
        x = X_flat[:, 0]
        y = X_flat[:, 1]
        plotted_labels = labels_flat
        title = title or "Flattened cluster labels"
    else:
        if labels.shape != X.shape[:2]:
            raise ValueError(
                "frame plotting requires labels with shape (n_particles, n_frames)."
            )
        if frame < 0 or frame >= X.shape[1]:
            raise IndexError("frame is out of bounds.")

        x = X[:, frame, 0]
        y = X[:, frame, 1]
        plotted_labels = labels[:, frame]
        title = title or f"Frame {frame} cluster labels"

    unique_labels = np.unique(plotted_labels)
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(unique_labels):
        mask = plotted_labels == label
        color = "lightgray" if label == -1 else cmap(idx % cmap.N)
        label_name = "unclassified" if label == -1 else f"cluster {int(label)}"
        plt.scatter(
            x[mask],
            y[mask],
            c=[color],
            label=label_name,
            s=15,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.2,
        )

    plt.title(title)
    plt.xlabel("feature 0")
    plt.ylabel("feature 1")
    plt.legend(loc="best", fontsize="small")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# ---------------- Fixtures ----------------


@pytest.fixture(scope="module")
def input_data_2d() -> np.ndarray:
    np.random.seed(42)
    cov1 = np.array([[0.05, 0.02], [0.02, 0.02]])
    cov2 = np.array([[0.1, -0.03], [-0.03, 0.05]])

    list_data_2d = []
    for _ in range(100):
        time_series = []
        for t in range(500):
            # Metastable states: salta tra (0, 0) e (1, 1) ogni 50 time steps
            state = (
                np.array([0.0, 0.0])
                if (t // 50) % 2 == 0
                else np.array([1.0, 1.0])
            )

            # Aggiungi noise gaussiano alternando tra cov1 e cov2
            cov = cov1 if (t // 50) % 2 == 0 else cov2
            noise = np.random.multivariate_normal([0, 0], cov)
            point = state + noise
            time_series.append(point)

        list_data_2d.append(np.array(time_series))

    return np.array(list_data_2d)


# ---------------- Tests ----------------


def test_onion(input_data_2d: np.ndarray):
    tau = 10

    # Test class interface
    on_cl = OnionClustering(tau=tau, bins=50)
    on_cl.fit(input_data_2d)

    # Test functional interface
    _ = onion_clustering(input_data_2d, tau=tau)

    # Check clustering output
    this_dir = Path(__file__).parent
    expected = np.load(this_dir / "output_multi_smooth/labels.npy")
    mask_0 = expected == 0
    mask_1 = expected == 1
    expected[mask_1] = 0
    expected[mask_0] = 1
    assert_array_equal(on_cl.labels, expected)


def test_debug_plot_cluster_labels(input_data_2d: np.ndarray):
    on_cl = OnionClustering(tau=10, bins=50)
    on_cl.fit(input_data_2d)

    _plot_cluster_labels(
        input_data_2d,
        on_cl.labels,
        frame=None,
        title="Debug cluster labels",
    )
    # Don't close plots so they can be seen
    # plt.close("all")


def test_wrong_input():
    on_cl = OnionClustering(tau=10)

    input_data1 = np.empty((0, 5, 5))  # empty array
    with pytest.raises(ValueError, match="Empty dataset."):
        on_cl.fit(input_data1)

    input_data1 = np.zeros((100, 1, 2))  # just one frame
    with pytest.raises(ValueError, match="n_frames = 1."):
        on_cl.fit(input_data1)

    input_data = np.random.rand(3, 4, 2) + 1j * np.random.rand(3, 4, 2)
    with pytest.raises(ValueError, match="Complex data not supported."):
        on_cl.fit(input_data)
