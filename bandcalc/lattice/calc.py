import numpy as np

from .generate import generate_k_path

hbar = 1
m = 1
V = 1

def eps_0(k, G):
    """
    Calculate the unpertubated energy for given k vector and reciprocal lattice vector

    :param k: k vector of the particle
    :param G: reciprocal lattice vector

    :type k: numpy.ndarray
    :type G: numpy.ndarray

    :rtype: float
    """

    return hbar**2/(2*m)*np.sum((k-G)**2, axis=1)

def matrix(k, lattice):
    """
    Construct the hamiltonian for any given k in a specified lattice

    :param k: k vector of the particle
    :param lattice: reciprocal lattice

    :type k: numpy.ndarray
    :type lattice: numpy.ndarray

    :rtype: numpy.ndarray
    """

    N = len(lattice)
    mat = np.ones((N, N))*V
    diag = np.diag(eps_0(k, lattice))
    np.fill_diagonal(mat, 0)
    mat += diag
    return mat

def calc_bandstructure(k_points, lattice, N):
    """
    Calculate the band structure of a lattice along a given k path with N samples

    :param k_points: k points
    :param lattice: reciprocal lattice
    :param N: number of samples

    :type k_points: numpy.ndarray
    :type lattice: numpy.ndarray
    :type N: int

    :rtype: numpy.ndarray
    """

    path = generate_k_path(k_points, N)
    eig_vals = np.array([np.linalg.eigvals(matrix(k, lattice)) for k in path])
    return eig_vals
