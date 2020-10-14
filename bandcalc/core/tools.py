import numpy as np

from bandcalc.constants import unit_prefixes

from functools import lru_cache

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

def find_k_order_delaunay_neighbours(point_index, delaunay_triangulation, k):
    """
    Find the nearest neighbours of a point with index *point_index*
    in a *delaunay_triangulation* up to the *k*-th shell of the point cloud.

    :param point_index: Index of the point of interest in the Delaunay triangulation
    :param delaunay_triangulation: Precalculated Delaunay triangulation of any point cloud
    :param k: Shell number

    :type point_index: int
    :type delaunay_triangulation: scipy.spatial.qhull.Delaunay
    :type k: int

    :rtype: list(int)
    """

    neighbours = newest_neighbours = find_nearest_delaunay_neighbours(point_index,
            delaunay_triangulation)
    for _ in range(k-1):
        newest_neighbours = set.union(*[
            find_nearest_delaunay_neighbours(neighbour, delaunay_triangulation)
            for neighbour in newest_neighbours]) - newest_neighbours
        neighbours = neighbours.union(newest_neighbours)
    
    # Always include starting point
    neighbours = neighbours.union({point_index})

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

