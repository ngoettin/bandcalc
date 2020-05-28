import numpy as np
import matplotlib.pyplot as plt

def plot_lattice(ax, lattice, *plotargs, **plotkargs):
    """
    Plot lattice on an axis

    :param ax: matplotlib axis
    :param lattice: lattice to plot

    :type ax: matplotlib.axes.Axes
    :type lattice: numpy.ndarray
    """

    ax.plot(lattice[:,0], lattice[:,1], *plotargs, **plotkargs)
    ax.axis("equal")

def plot_bandstructure(ax, bandstructure, k_point_names, *plotargs, **plotkargs):
    """
    Plot band structure on an axis

    :param ax: matplotlib axis
    :param bandstructure: bandstructure to plot
    :param k_point_names: names of the k points

    :type ax: matplotlib.axes.Axes
    :type bandstructure: numpy.ndarray
    :type k_point_names: list[str]
    """

    ax.plot(np.sort(bandstructure), *plotargs, **plotkargs)
    num_points = len(k_point_names)
    N = bandstructure.shape[0]
    ax.set_xticks([i*N/(num_points-1) for i in range(num_points)])
    ax.set_xticklabels(k_point_names)

def plot_k_path(ax, path, *plotargs, **plotkargs):
    """
    Plot k path on an axis

    :param ax: matplotlib axis
    :param path: path to plot

    :type ax: matplotlib.axes.Axes
    :type path: numpy.ndarray
    """

    ax.plot(path[:,0], path[:,1], *plotargs, **plotkargs)

