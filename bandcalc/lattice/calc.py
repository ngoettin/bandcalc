import numpy as np

from .generate import generate_k_path

hbar = 1
m = 1

def V(k):
    return 1

def eps_0(k, G):
    return hbar**2/(2*m)*np.sum((k-G)**2, axis=1)

def matrix(k, lattice):
    N = len(lattice)
    mat = np.ones((N, N))*V(k)
    diag = np.diag(eps_0(k, lattice))
    np.fill_diagonal(mat, 0)
    mat += diag
    return mat

def calc_bandstructure(k_points, lattice, N):
    path = generate_k_path(k_points, N)
    eig_vals = np.array([np.linalg.eigvals(matrix(k, lattice)) for k in path])
    return eig_vals
