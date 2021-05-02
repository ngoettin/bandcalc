import numpy as np
import scipy.spatial
import pyqtgraph as pg
import pyqtgraph.opengl as gl
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

    mpl_obj = ax.plot(lattice[:,0], lattice[:,1], *plotargs, **plotkargs)
    ax.axis("scaled")
    return mpl_obj

def plot_lattice_3d(ax, lattice, *plotargs, **plotkargs):
    """
    Plot lattice on an axis

    :param ax: matplotlib axis
    :param lattice: lattice to plot

    :type ax: mpl_toolkits.mplot3d.axes3d.Axes3D
    :type lattice: numpy.ndarray
    """

    mpl_obj = ax.scatter(lattice[:,0], lattice[:,1], lattice[:,2], *plotargs, **plotkargs)
    return mpl_obj

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

    mpl_obj = ax.plot(np.sort(bandstructure), *plotargs, **plotkargs)
    num_points = len(k_point_names)
    N = bandstructure.shape[0]
    ax.set_xticks([i*N/(num_points-1) for i in range(num_points)])
    ax.set_xticklabels(k_point_names)
    return mpl_obj

def plot_k_path(ax, path, *plotargs, **plotkargs):
    """
    Plot k path on an axis

    :param ax: matplotlib axis
    :param path: path to plot

    :type ax: matplotlib.axes.Axes
    :type path: numpy.ndarray
    """

    mpl_obj = ax.plot(path[:,0], path[:,1], *plotargs, **plotkargs)
    return mpl_obj

def plot_k_path_3d(ax, path, *plotargs, **plotkargs):
    """
    Plot k path on an axis

    :param ax: matplotlib axis
    :param path: path to plot

    :type ax: mpl_toolkits.mplot3d.axes3d.Axes3D
    :type path: numpy.ndarray
    """

    mpl_obj = ax.plot(path[:,0], path[:,1], path[:,2], *plotargs, **plotkargs)
    return mpl_obj

def plot_moire_potential(ax, grid, moire_potential, *plotargs, **plotkargs):
    """
    Plot a moire potential on a regular grid

    :param ax: matplotlib axis
    :param grid: grid to plot on
    :param moire_potential: moire potential

    :type ax: matplotlib.axes.Axes
    :type grid: numpy.ndarray
    :type moire_potential: numpy.ndarray
    """

    mpl_obj = ax.contourf(grid[0], grid[1], np.real(moire_potential), *plotargs, **plotkargs)
    ax.axis("scaled")
    return mpl_obj

def plot_wave_function(ax, grid, wave_function, *plotargs, **plotkargs):
    """
    Plot a wave funtion on a regular grid.

    :param ax: matplotlib axis
    :param grid: grid to plot on
    :param wave_function: wave function to plot

    :type ax: matplotlib.axes.Axes
    :type grid: numpy.ndarray
    :type wave_function: numpy.ndarray
    """

    mpl_obj = ax.imshow(wave_function, *plotargs,
            extent=[grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()],
            origin="lower", **plotkargs)
    ax.axis("scaled")
    return mpl_obj

def plot_matrix(matrix):
    """
    Convenience function for plotting a matrix

    :param matrix: matrix to plot

    :type matrix: numpy.ndarray
    """

    _, ax = plt.subplots()
    ax.pcolor(matrix, ec="#AAA", lw=0.5)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(matrix.shape[1])+0.5)
    ax.set_yticks(np.arange(matrix.shape[0])+0.5)
    ax.set_xticklabels(range(matrix.shape[1]))
    ax.set_yticklabels(range(matrix.shape[0]))
    ax.axis("scaled")
    ax.invert_yaxis()
    ax.format_coord = lambda x, y: f"({int(y)}, {int(x)}): {matrix[int(x), int(y)]:.2f}"

def plot_trisurface_3d(x, y, z, faces="Delaunay", autoscale_z=True):
    """
    Plots any triangular surface with OpenGL acceleration.
    Use this for surface plots with many data points.

    :param x: x coordinates. Should be a 1D array
    :param y: y coordinates. Should be a 1D array
    :param z: z coordinates. Should be a 1D array
    :param faces: calculated faces for the vertices. Default will use Delaunay
        to automatically get surfaces.

    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :type z: numpy.ndarray
    :type faces: str | numpy.ndarray
    """

    app = pg.mkQApp()
    view = gl.GLViewWidget()

    if autoscale_z:
        z = z/(z.max()-z.min())*max(x.max(), y.max())

    vertices = np.vstack([x, y, z]).T
    if faces == "Delaunay":
        triangulation = scipy.spatial.Delaunay(np.vstack([x, y]).T) #pylint: disable=E1101
        faces = triangulation.simplices

    meshdata = gl.MeshData(vertexes=vertices, faces=faces)
    mesh = gl.GLMeshItem(meshdata=meshdata, shader="normalColor",
            smooth=True, computeNormals=True)

    view.addItem(mesh)
    view.show()
    app.exec_()
