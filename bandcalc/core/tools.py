import numpy as np

from bandcalc.constants import unit_prefixes

from functools import lru_cache
from scipy.spatial import Voronoi, ConvexHull, KDTree # pylint: disable=E0611

def get_unit_prefix(data):
    """
    Suggest a unit prefix for given *data*, to make it more readable

    :param data: find unit prefix for this data

    :type data: numpy.ndarray

    :rtype: tuple(numpy.ndarray, str)
    """

    max = np.abs(data).max()

    if max == 0:
        return (data, "")

    exponent = int(np.floor(np.log(max)/np.log(1000)))*3

    if exponent in unit_prefixes:
        data *= 10**(-exponent)
        prefix = unit_prefixes[exponent]
        return (data, prefix)

    return (data, "")

@lru_cache(maxsize=5_000_000)
def find_nearest_delaunay_neighbours(point_index, delaunay_triangulation):
    """
    Find the nearest neighbours of a point with index *point_index*
    in a *delaunay_triangulation*.

    :param point_index: Index of the point of interest in the Delaunay triangulation
    :param delaunay_triangulation: Precalculated Delaunay triangulation of any point cloud

    :type point_index: int
    :type delaunay_triangulation: scipy.spatial.qhull.Delaunay

    :rtype: list(int)
    """
    
    indexptr, indices = delaunay_triangulation.vertex_neighbor_vertices
    return set(indices[indexptr[point_index]:indexptr[point_index+1]])

def find_k_order_delaunay_neighbours(point_index, delaunay_triangulation, k, only_k_shell=False,
        include_point_index=True):
    """
    Find the nearest neighbours of a point with index *point_index*
    in a *delaunay_triangulation* up to the *k*-th shell of the point cloud.

    :param point_index: Index of the point of interest in the Delaunay triangulation
    :param delaunay_triangulation: Precalculated Delaunay triangulation of any point cloud
    :param k: Shell number
    :param only_k_shell: If True, return only the k-order neighbours and not all lower orders
    :param include_point_index: Whether to include the *point_index* or not.

    :type point_index: int
    :type delaunay_triangulation: scipy.spatial.qhull.Delaunay
    :type k: int
    :type only_k_shell: bool
    :type include_point_index: bool

    :rtype: list(int)
    """

    if only_k_shell and k<1:
        raise Exception("Can't use only_k_shell=True if k is smaller than 1.")

    if only_k_shell and include_point_index:
        raise Exception("Can't include point_index if only the k-th shell should "\
                "be returned")

    neighbours = newest_neighbours = {point_index}
    for _ in range(k):
        newest_neighbours = set.union(*[
            find_nearest_delaunay_neighbours(neighbour, delaunay_triangulation)
            for neighbour in newest_neighbours]) - newest_neighbours
        neighbours = neighbours.union(newest_neighbours)
    
    if include_point_index:
        neighbours = neighbours.union({point_index})
    else:
        neighbours = neighbours - {point_index}

    if only_k_shell:
        neighbours = neighbours - set(find_k_order_delaunay_neighbours(
                point_index, delaunay_triangulation, k-1,
                only_k_shell=False, include_point_index=False))

    return list(neighbours)

def find_vector_index(lattice, vector):
    """
    Finds the index of the first occurence of a vector in a lattice, if it exists.

    :param lattice: Lattice, which should contain the vector of interest
    :param vector: The vector of interest

    :type lattice: numpy.ndarray
    :type vector: numpy.ndarray

    :rtype: int
    """

    try:
        vec_index = np.where(np.all(lattice == vector, axis=1))[0][0]
    except IndexError:
        vec_index = None
    return vec_index

def get_volume_element_regular_grid(grid_points):
    """
    Calculate unit cell volume

    :param grid_points: points of the grid

    :type grid_points: numpy.ndarray

    :rtype: float
    """

    # Get point somewhere in the center of the grid to (most likely) ensure
    # a closed voronoi cell
    tree = KDTree(grid_points)
    center_point_ind = tree.query(np.average(grid_points, axis=0))[1]

    # Get the point region of the central grid point, find the corresponding
    # vertex indices and get the vertex points
    voronoi = Voronoi(grid_points)
    innermost_cell_points = voronoi.vertices[voronoi.regions[voronoi.point_region[center_point_ind]]]

    # Calculate size of innermost cell
    volume = ConvexHull(innermost_cell_points).volume
    return volume

def integrate_2d_func_regular_grid(func_vals, grid_points):
    """
    Integrate precalculated function values over a regular grid.
    Area elements will be calculated as the size of a Voronoi cell.

    :param func_vals: *n* function values
    :param grid_points: (*n*, 2) array of grid points

    :type func_vals: numpy.ndarray
    :type grid_points: numpy.ndarray

    :rtype: float
    """

    dA = get_volume_element_regular_grid(grid_points)

    # Integrate
    integral = np.sum(func_vals)*dA
    return integral
