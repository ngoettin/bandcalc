import itertools

import numpy as np

import scipy.interpolate

def group_lattice_vectors_by_length(lattice):
    """
    Group an array of vectors by length

    :param lattice: vectors to group

    :type lattice: numpy.ndarray

    :rtype: dict[float, numpy.ndarray]
    """

    lattice = np.array(sorted(lattice, key=lambda vec: np.dot(vec, vec)))
    ordered_lattice = {}
    for vec in lattice:
        length = np.round(np.dot(vec, vec), 2)
        try:
            ordered_lattice[length].append(vec)
        except KeyError:
            ordered_lattice[length] = [vec]
    ordered_lattice = {k: np.array(v) for k,v in ordered_lattice.items()}
    return ordered_lattice

def generate_lattice(lattice_basis, size):
    """
    Combine lattice basis vectors *size* times to generate a lattice

    :param lattice_basis: basis vectors of the lattice
    :param size: maximum coefficient size

    :type lattice_basis: numpy.ndarray
    :type size: int

    :rtype: numpy.ndarray
    """

    dimension = len(lattice_basis)
    combinations = np.array(list(itertools.product(range(-size, size+1), repeat=dimension)))
    lattice = np.matmul(lattice_basis.T, combinations.T)
    return lattice.T

def generate_lattice_by_shell(lattice_basis, shell):
    """
    Generate lattice from basis vectors with *shell* shells around the zero vector

    :param lattice_basis: basis vectors of the lattice
    :param shell: number of shells

    :type lattice_basis: numpy.ndarray
    :type shell: int

    :rtype: numpy.ndarray
    """

    uncut_lattice = generate_lattice(lattice_basis, shell)
    ordered_lattice = group_lattice_vectors_by_length(uncut_lattice)
    try:
        shortest_vector = list(ordered_lattice.keys())[1]
    except IndexError:
        shortest_vector = np.array([0,0])
    for key in ordered_lattice.copy().keys():
        if key>shortest_vector*shell**2:
            del ordered_lattice[key]
    return np.vstack(list(ordered_lattice.values()))

def generate_k_path(points, N):
    """
    Interpolate between *points* with *N* steps

    :param points: points to interpolate between
    :param N: sample number

    :type points: numpy.ndarray
    :type N: int

    :rtype: numpy.ndarray
    """

    num_points = len(points)
    path = scipy.interpolate.griddata(np.arange(num_points), points, np.linspace(0, num_points-1, N))
    return path

