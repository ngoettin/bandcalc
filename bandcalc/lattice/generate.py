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

def generate_rotation_matrix(angle, degrees=True):
    """
    Generate a 2D rotation matrix for given *angle*

    :param angle: the angle to rotate by
    :param degrees: if angle is given in degrees or not (radians)

    :type angle: float
    :type degrees: bool

    :rtype: numpy.ndarray
    """

    if degrees:
        angle *= np.pi/180
    return np.array([[np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])

def rotate_lattice(lattice, angle, degrees=True):
    """
    Rotate a 2D lattice by *angle*

    :param lattice: lattice to rotate
    :param angle: the angle to rotate by
    :param degrees: if angle is given in degrees or not (radians)

    :type lattice: numpy.ndarray
    :type angle: float
    :type degrees: bool

    :rtype: numpy.ndarray
    """

    rotation_matrix = generate_rotation_matrix(angle)
    rotated_lattice = np.matmul(rotation_matrix, lattice.T).T
    return rotated_lattice

def generate_moire_lattice_by_shell(lattice_basis1, lattice_basis2, twist_angle, shell):
    """
    Generate a twisted moire lattice from two sets of basis vectors

    :param lattice_basis1: lattice basis of first lattice
    :param lattice_basis2: lattice basis of second lattice
    :param twist_angle: twist angle
    :shell: number of shells to calculate

    :type lattice_basis1: numpy.ndarray
    :type lattice_basis2: numpy.ndarray
    :type twist_angle: float
    :type shell: int

    :rtype: numpy.ndarray
    """

    lattice1 = generate_lattice_by_shell(lattice_basis1, shell)
    lattice2 = generate_lattice_by_shell(lattice_basis2, shell)
    lattice2 = rotate_lattice(lattice2, twist_angle)
    return np.stack([lattice1, lattice2])

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

