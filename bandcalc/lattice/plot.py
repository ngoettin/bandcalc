import numpy as np
import matplotlib.pyplot as plt

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Source: https://stackoverflow.com/a/31364297
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

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

def plot_lattice_3d(ax, lattice, *plotargs, **plotkargs):
    """
    Plot lattice on an axis

    :param ax: matplotlib axis
    :param lattice: lattice to plot

    :type ax: mpl_toolkits.mplot3d.axes3d.Axes3D
    :type lattice: numpy.ndarray
    """

    ax.scatter(lattice[:,0], lattice[:,1], lattice[:,2], *plotargs, **plotkargs)
#    ax.axis("equal")

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

def plot_k_path_3d(ax, path, *plotargs, **plotkargs):
    """
    Plot k path on an axis

    :param ax: matplotlib axis
    :param path: path to plot

    :type ax: mpl_toolkits.mplot3d.axes3d.Axes3D
    :type path: numpy.ndarray
    """

    ax.plot(path[:,0], path[:,1], path[:,2], *plotargs, **plotkargs)
