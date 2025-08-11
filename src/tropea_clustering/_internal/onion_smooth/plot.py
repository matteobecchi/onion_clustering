"""Auxiliary functions for plotting the results of onion-clustering.

* Author: Becchi Matteo <bechmath@gmail.com>
* Date: November 28, 2024
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from tropea_clustering._internal.onion_smooth.functions import gaussian
from tropea_clustering._internal.onion_smooth.main import StateUni
from tropea_clustering._internal.onion_smooth.main_2d import StateMulti

COLORMAP = "viridis"


def plot_output_uni(
    title: Path,
    input_data: NDArray[np.float64],
    state_list: list[StateUni],
):
    """Plots clustering output with Gaussians and thresholds.

    Parameters
    ----------
    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    input_data : ndarray of shape (n_particles, n_frames)
        The input data array.

    state_list : list[StateUni]
        The list of the cluster states.

    Example
    -------
    Here's an example of the output:

    .. image:: ../_static/images/uni_Fig1.png
        :alt: Example Image
        :width: 600px

    The left planel shows the input time-series data, with the backgound
    colored according to the thresholds between the clusters. The left panel
    shows the cumulative data distribution, and the Gaussians fitted to the
    data, corresponding to the identified clusters.
    """
    n_particles, t_steps = input_data.shape

    flat_m = input_data.flatten()
    counts, bins = np.histogram(flat_m, bins=100, density=True)
    bins -= (bins[1] - bins[0]) / 2
    counts *= flat_m.size

    fig, axes = plt.subplots(
        1,
        2,
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=(9, 4.8),
    )

    axes[1].stairs(
        counts, bins, fill=True, orientation="horizontal", alpha=0.5
    )

    palette = []
    n_states = len(state_list)
    cmap = plt.get_cmap(COLORMAP, n_states + 1)
    for i in range(1, cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    time = np.linspace(0, t_steps - 1, t_steps)

    step = 1
    if input_data.size > 1e6:
        step = 10
    for mol in input_data[::step]:
        axes[0].plot(
            time,
            mol,
            c="xkcd:black",
            ms=0.1,
            lw=0.1,
            alpha=0.5,
            rasterized=True,
        )

    for state_id, state in enumerate(state_list):
        attr = state.get_attributes()
        popt = [attr["mean"], attr["sigma"], attr["area"]]
        axes[1].plot(
            gaussian(np.linspace(bins[0], bins[-1], 1000), *popt),
            np.linspace(bins[0], bins[-1], 1000),
            color=palette[state_id],
        )

    style_color_map = {
        0: ("--", "xkcd:black"),
        1: ("--", "xkcd:blue"),
        2: ("--", "xkcd:red"),
    }

    time2 = np.linspace(
        time[0] - 0.05 * (time[-1] - time[0]),
        time[-1] + 0.05 * (time[-1] - time[0]),
        100,
    )
    for state_id, state in enumerate(state_list):
        th_inf = state.get_attributes()["th_inf"]
        th_sup = state.get_attributes()["th_sup"]
        linestyle, color = style_color_map.get(th_inf[1], ("-", "xkcd:black"))
        axes[1].hlines(
            th_inf[0],
            xmin=0.0,
            xmax=np.amax(counts),
            linestyle=linestyle,
            color=color,
        )
        axes[0].fill_between(
            time2,
            th_inf[0],
            th_sup[0],
            color=palette[state_id],
            alpha=0.25,
        )
    axes[1].hlines(
        state_list[-1].get_attributes()["th_sup"][0],
        xmin=0.0,
        xmax=np.amax(counts),
        linestyle=linestyle,
        color="black",
    )

    # Set plot titles and axis labels
    axes[0].set_ylabel("Signal")
    axes[0].set_xlabel(r"Time [frame]")
    axes[1].set_xticklabels([])

    fig.savefig(title, dpi=600)


def plot_one_trj_uni(
    title: Path,
    example_id: int,
    input_data: NDArray[np.float64],
    labels: NDArray[np.int64],
):
    """Plots the colored trajectory of one example particle.

    Unclassified data points are colored with the darkest color.

    Parameters
    ----------

    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    example_id : int
        The ID of the selected particle.

    input_data : ndarray of shape (n_particles, n_frames)
        The input data array.

    labels : ndarray of shape (n_particles, n_frames)
        The output of Onion Clustering.

    Example
    -------
    Here's an example of the output:

    .. image:: ../_static/images/uni_Fig2.png
        :alt: Example Image
        :width: 600px

    The datapoints are colored according to the cluster they have been
    assigned.
    """
    n_particles, n_frames = input_data.shape
    time = np.linspace(0, n_frames - 1, n_frames)

    fig, axes = plt.subplots()
    unique_labels = np.unique(labels)
    # If there are no assigned window, we still need the "-1" state
    # for consistency:
    if -1 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, -1)

    cmap = plt.get_cmap(COLORMAP, unique_labels.size)
    color = labels[example_id] + 1
    axes.plot(time, input_data[example_id], c="black", lw=0.1)

    axes.scatter(
        time,
        input_data[example_id],
        c=color,
        cmap=cmap,
        vmin=0,
        vmax=unique_labels.size - 1,
        s=1.0,
    )

    # Add title and labels to the axes
    fig.suptitle(f"Example particle: ID = {example_id}")
    axes.set_xlabel("Time [frame]")
    axes.set_ylabel("Signal")

    fig.savefig(title, dpi=600)


def plot_state_populations(
    title: Path,
    labels: NDArray[np.int64],
):
    """
    Plot the populations of clusters over time.

    For each trajectory frame, plots the fraction of the population of each
    cluster. In the legend, "ENV0" refers to the unclassified data.

    Parameters
    ----------
    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    labels : ndarray of shape (n_particles, n_frames)
        The output of Onion Clustering.

    Example
    -------
    Here's an example of the output:

    .. image:: ../_static/images/uni_Fig4.png
        :alt: Example Image
        :width: 600px
    """
    n_particles, n_frames = labels.shape
    unique_labels = np.unique(labels)
    if -1 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, -1)

    list_of_populations = []
    for label in unique_labels:
        population = np.sum(labels == label, axis=0)
        list_of_populations.append(population / n_particles)

    palette = []
    cmap = plt.get_cmap(COLORMAP, unique_labels.size)
    for i in range(cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    fig, axes = plt.subplots()
    time = range(n_frames)
    for label, pop in enumerate(list_of_populations):
        axes.plot(time, pop, label=f"ENV{label}", color=palette[label])
    axes.set_xlabel(r"Time [frame]")
    axes.set_ylabel(r"Population fraction")
    axes.legend()

    fig.savefig(title, dpi=600)


def plot_sankey(
    title: Path,
    labels: NDArray[np.int64],
    tmp_frame_list: list[int] | NDArray[np.int64],
):
    """
    Plots the Sankey diagram at the desired frames.

    .. warning::
        This function requires the python package Kaleido, and uses plotly
        instead of matplotlib.pyplot. For this reason is deprecated and not
        supported since tropea-clustering 2.0.0.

    Parameters
    ----------

    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    labels : ndarray of shape (n_particles, n_frames)
        The output of the clustering algorithm.

    tmp_frame_list : list[int] | NDArray[np.int64]
        The list of frames at which we want to plot the Sankey.

    Example
    -------
    Here's an example of the output:

    .. image:: ../_static/images/uni_Fig5.png
        :alt: Example Image
        :width: 600px

    For each of the selected frames, the colored bars width is proportional
    to each cluster population. The gray bands' witdh are proportional to
    the number of data points moving from one cluster to the other between the
    selected frames. State "-1" refers to the unclassified data.
    """
    frame_list = np.array(tmp_frame_list)
    unique_labels = np.unique(labels)

    if -1 not in unique_labels:
        unique_labels = np.insert(unique_labels, 0, -1)
    n_states = unique_labels.size

    source = np.empty((frame_list.size - 1) * n_states**2)
    target = np.empty((frame_list.size - 1) * n_states**2)
    value = np.empty((frame_list.size - 1) * n_states**2)

    count = 0
    tmp_label = []

    # Loop through the frame_list and calculate the transition matrix
    # for each time window.
    for i, t_0 in enumerate(frame_list[:-1]):
        # Calculate the time jump for the current time window.
        t_jump = frame_list[i + 1] - frame_list[i]

        trans_mat = np.zeros((n_states, n_states))

        # Iterate through the current time window and increment
        # the transition counts in trans_mat
        for label in labels:
            trans_mat[label[t_0] + 1][label[t_0 + t_jump] + 1] += 1

        # Store the source, target, and value for the Sankey diagram
        # based on trans_mat
        for j, row in enumerate(trans_mat):
            for k, elem in enumerate(row):
                source[count] = j + i * n_states
                target[count] = k + (i + 1) * n_states
                value[count] = elem
                count += 1

        # Create node labels
        for j in unique_labels:
            tmp_label.append(f"State {j}")

    state_label = np.array(tmp_label).flatten()

    # Generate a color palette for the Sankey diagram.
    palette = []
    cmap = plt.get_cmap(COLORMAP, n_states)
    for i in range(cmap.N):
        rgba = cmap(i)
        palette.append(rgb2hex(rgba))

    # Tile the color palette to match the number of frames.
    color = np.tile(palette, frame_list.size)

    # Create dictionaries to define the Sankey diagram nodes and links.
    node = {"label": state_label, "pad": 30, "thickness": 20, "color": color}
    link = {"source": source, "target": target, "value": value}

    # Create the Sankey diagram using Plotly.
    sankey_data = go.Sankey(link=link, node=node, arrangement="perpendicular")
    fig = go.Figure(sankey_data)

    # Add the title with the time information.
    fig.update_layout(title=f"Frames: {frame_list}")

    fig.write_image(title, scale=5.0)


def plot_time_res_analysis(
    title: Path,
    tra: NDArray[np.float64],
):
    """
    Plots the results of clustering at different time resolutions.

    Parameters
    ----------
    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    tra : ndarray of shape (delta_t_values, 3)
        tra[j][0] must contain the j-th value used as delta_t;
        tra[j][1] must contain the corresponding number of states;
        tra[j][2] must contain the corresponding unclassified fraction.

    Example
    -------
    Here's an example of the output:

    .. image:: ../_static/images/uni_Fig6.png
        :alt: Example Image
        :width: 600px

    For each of the analyzed time resolutions, the blue curve shows the number
    of identified clusters (not including the unclassified data); the orange
    line shows the fraction of unclassififed data.
    """
    fig, ax = plt.subplots()
    ax.plot(tra[:, 0], tra[:, 1], marker="o")
    ax.set_xlabel(r"Time resolution $\Delta t$ [frame]")
    ax.set_ylabel(r"# environments", weight="bold", c="#1f77b4")
    ax.set_xscale("log")
    ax.set_ylim(-0.2, np.max(tra[:, 1]) + 0.2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_r = ax.twinx()
    ax_r.plot(tra[:, 0], tra[:, 2], marker="o", c="#ff7f0e")
    ax_r.set_ylabel("Unclassified fraction", weight="bold", c="#ff7f0e")
    ax_r.set_ylim(-0.02, 1.02)
    fig.savefig(title, dpi=600)


def plot_pop_fractions(
    title: Path,
    list_of_pop: list[list[float]],
    tra: NDArray[np.float64],
):
    """
    Plot, for every time resolution, the populations of the clusters.

    Parameters
    ----------
    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    list_of_pop : list[list[float]]
        For every delta_t, this is the list of the populations of all the
        states (the first one is the unclassified data points).

    tra : ndarray of shape (delta_t_values, 3)
        tra[j][0] must contain the j-th value used as delta_t;
        tra[j][1] must contain the corresponding number of states;
        tra[j][2] must contain the corresponding unclassified fraction.

    Example
    -------
    Here's an example of the output:

    .. image:: ../_static/images/uni_Fig7.png
        :alt: Example Image
        :width: 600px

    For each time resolution analysed, the bars show the fraction of data
    points classified in each cluster. Clusters are ordered according to the
    value of their Gaussian's mean; the bottom cluster is always the
    unclassified data points.
    """
    # Pad the lists in list_of_pop to ensure they all have the same length
    max_num_of_states = np.max([len(pop_list) for pop_list in list_of_pop])
    for pop_list in list_of_pop:
        while len(pop_list) < max_num_of_states:
            pop_list.append(0.0)

    pop_array = np.array(list_of_pop)

    fig, axes = plt.subplots()

    time = tra[:, 0]
    bottom = np.zeros(len(pop_array))
    width = time / 2 * 0.5

    for _, state in enumerate(pop_array.T):
        _ = axes.bar(time, state, width, bottom=bottom, edgecolor="black")
        bottom += state

    axes.set_xlabel(r"Time resolution $\Delta t$ [frames]")
    axes.set_ylabel(r"Populations fractions")
    axes.set_xscale("log")

    fig.savefig(title, dpi=600)


def plot_output_multi(
    title: Path,
    input_data: NDArray[np.float64],
    state_list: list[StateMulti],
    labels: NDArray[np.int64],
):
    """
    Plot a cumulative figure showing trajectories and identified states.

    Parameters
    ----------
    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    input_data : ndarray of shape (n_particles, n_frames, n_features)
        The input data array.

    state_list : list[StateMulti]
        The list of the cluster states.

    labels : ndarray of shape (n_particles, n_frames)
        The output of the clustering algorithm.

    Example
    -------
    .. image:: ../_static/images/multi_Fig1.png
        :alt: Example Image
        :width: 600px

    All the data are plotted, colored according to the cluster thay have been
    assigned to. The clusters are shown as black ellipses, whose orizontal and
    vertical axis length is given by the standard deviation of the Gaussians
    corresponding to the cluster. Unclassififed data points are colored in
    purple.
    """
    n_states = len(state_list) + 1
    tmp = plt.get_cmap(COLORMAP, n_states)
    colors_from_cmap = tmp(np.arange(0, 1, 1 / n_states))
    colors_from_cmap[-1] = tmp(1.0)

    if input_data.shape[2] == 3:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        dir0 = [0, 0, 1]
        dir1 = [1, 2, 2]
        ax0 = [0, 0, 1]
        ax1 = [0, 1, 0]

        id_max, id_min = 0, 0
        for idx, mol in enumerate(input_data):
            if np.max(mol) == np.max(input_data):
                id_max = idx
            if np.min(mol) == np.min(input_data):
                id_min = idx

        for k in range(3):
            d_0 = dir0[k]
            d_1 = dir1[k]
            a_0 = ax0[k]
            a_1 = ax1[k]

            # Plot the individual trajectories
            line_w = 0.05
            max_t = labels.shape[1]
            step = 5 if input_data.size > 1000000 else 1

            for i, mol in enumerate(input_data[::step]):
                print(mol.shape)
                ax[a_0][a_1].plot(
                    mol[:, d_0],  # type: ignore  # mol is 2D, mypy can't see it
                    mol[:, d_1],  # type: ignore  # mol is 2D, mypy can't see it
                    c="black",
                    lw=line_w,
                    rasterized=True,
                    zorder=0,
                )
                color_list = labels[i * step] + 1
                ax[a_0][a_1].scatter(
                    mol.T[d_0],  # type: ignore  # mol is 2D, mypy can't see it
                    mol.T[d_1],  # type: ignore  # mol is 2D, mypy can't see it
                    c=color_list,
                    cmap=COLORMAP,
                    vmin=0,
                    vmax=n_states - 1,
                    s=0.5,
                    rasterized=True,
                )

                color_list = labels[id_min] + 1
                ax[a_0][a_1].plot(
                    input_data[id_min].T[d_0],
                    input_data[id_min].T[d_1],
                    c="black",
                    lw=line_w,
                    rasterized=True,
                    zorder=0,
                )
                ax[a_0][a_1].scatter(
                    input_data[id_min].T[d_0],
                    input_data[id_min].T[d_1],
                    c=color_list,
                    cmap=COLORMAP,
                    vmin=0,
                    vmax=n_states - 1,
                    s=0.5,
                    rasterized=True,
                )
                color_list = labels[id_max] + 1
                ax[a_0][a_1].plot(
                    input_data[id_max].T[d_0],
                    input_data[id_max].T[d_1],
                    c="black",
                    lw=line_w,
                    rasterized=True,
                    zorder=0,
                )
                ax[a_0][a_1].scatter(
                    input_data[id_max].T[d_0],
                    input_data[id_max].T[d_1],
                    c=color_list,
                    cmap=COLORMAP,
                    vmin=0,
                    vmax=n_states - 1,
                    s=0.5,
                    rasterized=True,
                )

            # Set plot titles and axis labels
            ax[a_0][a_1].set_xlabel(f"Signal {d_0}")
            ax[a_0][a_1].set_ylabel(f"Signal {d_1}")

        ax[1][1].axis("off")
        fig.savefig(title, dpi=600)
        plt.close(fig)

    elif input_data.shape[2] == 2:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the individual trajectories
        id_max, id_min = 0, 0
        for idx, mol in enumerate(input_data):
            if np.max(mol) == np.max(input_data):
                id_max = idx
            if np.min(mol) == np.min(input_data):
                id_min = idx

        line_w = 0.05
        max_t = labels.shape[1]
        m_resized = input_data[:, :max_t:, :]
        step = 5 if m_resized.size > 1000000 else 1

        for i, mol in enumerate(m_resized[::step]):
            ax.plot(
                mol.T[0],  # type: ignore  # mol is 2D, mypy can't see it
                mol.T[1],  # type: ignore  # mol is 2D, mypy can't see it
                c="black",
                lw=line_w,
                rasterized=True,
                zorder=0,
            )
            color_list = labels[i * step] + 1
            ax.scatter(
                mol.T[0],  # type: ignore  # mol is 2D, mypy can't see it
                mol.T[1],  # type: ignore  # mol is 2D, mypy can't see it
                c=color_list,
                cmap=COLORMAP,
                vmin=0,
                vmax=n_states - 1,
                s=0.5,
                rasterized=True,
            )

        color_list = labels[id_min] + 1
        ax.plot(
            m_resized[id_min].T[0],
            m_resized[id_min].T[1],
            c="black",
            lw=line_w,
            rasterized=True,
            zorder=0,
        )
        ax.scatter(
            m_resized[id_min].T[0],
            m_resized[id_min].T[1],
            c=color_list,
            cmap=COLORMAP,
            vmin=0,
            vmax=n_states - 1,
            s=0.5,
            rasterized=True,
        )
        color_list = labels[id_max] + 1
        ax.plot(
            m_resized[id_max].T[0],
            m_resized[id_max].T[1],
            c="black",
            lw=line_w,
            rasterized=True,
            zorder=0,
        )
        ax.scatter(
            m_resized[id_max].T[0],
            m_resized[id_max].T[1],
            c=color_list,
            cmap=COLORMAP,
            vmin=0,
            vmax=n_states - 1,
            s=0.5,
            rasterized=True,
        )

        # Plot the Gaussian distributions of states
        for state in state_list:
            att = state.get_attributes()
            width, height, angle = state.get_boundaries()
            ellipse = Ellipse(
                xy=att["mean"],
                width=width,
                height=height,
                angle=angle,
                color="black",
                fill=False,
            )
            ax.add_patch(ellipse)

        # Set plot titles and axis labels
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        fig.savefig(title, dpi=600)


def plot_one_trj_multi(
    title: Path,
    example_id: int,
    input_data: NDArray[np.float64],
    labels: NDArray[np.int64],
):
    """Plots the colored trajectory of an example particle.

    Parameters
    ----------
    title : pathlib.Path
        The path of the .png file the figure will be saved as.

    example_id : int
        The ID of the selected particle.

    input_data : ndarray of shape (n_particles, n_frames, n_features)
        The input data array.

    labels : ndarray of shape (n_particles, n_frames)
        The output of the clustering algorithm.

    Example
    -------
    Here's an example of the output:

    .. image:: ../_static/images/multi_Fig2.png
        :alt: Example Image
        :width: 600px

    The datapoints are colored according to the cluster they have been
    assigned to.
    """
    # Get the signal of the example particle
    sig_x = input_data[example_id].T[0]
    sig_y = input_data[example_id].T[1]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a colormap to map colors to the labels
    cmap = plt.get_cmap(
        COLORMAP,
        int(np.max(np.unique(labels)) - np.min(np.unique(labels)) + 1),
    )
    color = labels[example_id]
    ax.plot(sig_x, sig_y, c="black", lw=0.1)

    ax.scatter(
        sig_x,
        sig_y,
        c=color,
        cmap=cmap,
        vmin=float(np.min(np.unique(labels))),
        vmax=float(np.max(np.unique(labels))),
        s=1.0,
        zorder=10,
    )

    # Set plot titles and axis labels
    fig.suptitle(f"Example particle: ID = {example_id}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    fig.savefig(title, dpi=600)


def color_trj_from_xyz(
    trj_path: str,
    labels: np.ndarray,
    n_particles: int,
    tau_window: int,
):
    """
    Saves a colored .xyz file ('colored_trj.xyz') in the working directory.

    Warning
    -------
    This function is WIP.

    Parameters
    ----------

    trj_path : str
        The path to the input .xyz trajectory.

    labels : np.ndarray (n_particles * n_windows,)
        The output of the clustering algorithm.

    n_particles : int
        The number of particles in the system.

    tau_window : int
        The length of the signal windows.

    Notes
    -----
    In the input file, the (x, y, z) coordinates of the particles need to be
    stored in the second, third and fourth column respectively.
    """
    if os.path.exists(trj_path):
        with open(trj_path, "r", encoding="utf-8") as in_file:
            tmp = [line.strip().split() for line in in_file]

        tmp_labels = labels.reshape((n_particles, -1))
        all_the_labels = np.repeat(tmp_labels, tau_window, axis=1) + 1
        total_time = int(labels.shape[0] / n_particles) * tau_window
        nlines = (n_particles + 2) * total_time
        tmp = tmp[:nlines]

        with open("colored_trj.xyz", "w+", encoding="utf-8") as out_file:
            i = 0
            for j in range(total_time):
                print(tmp[i][0], file=out_file)
                print("Properties=species:S:1:pos:R:3", file=out_file)
                for k in range(n_particles):
                    print(
                        all_the_labels[k][j],
                        tmp[i + 2 + k][1],
                        tmp[i + 2 + k][2],
                        tmp[i + 2 + k][3],
                        file=out_file,
                    )
                i += n_particles + 2
    else:
        raise ValueError(f"ValueError: {trj_path} not found.")
