import cmath

import cupy as cp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay #pylint: disable=E0611
from scipy.constants import physical_constants
from numba import cuda

import bandcalc
from bandcalc.constants import lattice_constants

potential = "MoS2"
angle = 2
energy_level = 1
shells = 1
a = lattice_constants["MoS2"]*1e9
N = 1000
mass = 0.35*physical_constants["electron mass"][0]

# Reciprocal lattice vectors
b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
b = np.vstack([b1, b2])

# Reciprocal moire basis vectors
rec_m = b-bandcalc.rotate_lattice(b, angle)

# Complete reciprocal moire lattice
rec_moire_lattice = bandcalc.generate_lattice_by_shell(rec_m, shells)

# Real space moire basis vectors
m = bandcalc.generate_reciprocal_lattice_basis(rec_m)

# Real space moire lattice vectors
moire_lattice = bandcalc.generate_lattice_by_shell(m, shells)

if potential == "MoS2":
    # Moire potential coefficients
    V = 6.6*1e-3*np.exp(-1j*94*np.pi/180) # in eV
    Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])
    potential_matrix = bandcalc.calc_potential_matrix_from_coeffs(rec_moire_lattice, Vj)
elif potential == "off":
    potential_matrix = bandcalc.calc_potential_matrix(rec_moire_lattice)

## Calculate eigenstates for every k point in the MBZ
k_points = bandcalc.generate_monkhorst_pack_set(m, 130).astype(np.float32)
hamiltonian = bandcalc.calc_hamiltonian(rec_moire_lattice, potential_matrix, mass)

c_alpha = []
for k_point in k_points:
    ham = hamiltonian(k_point)
    eig_vals, eig_vecs = np.linalg.eig(ham)
    eig_vecs = eig_vecs[:, np.argsort(eig_vals)]
    c_alpha.append(np.real(eig_vecs[:, 0]))
c_alpha = np.array(c_alpha)

## Transfer all variables to GPU
rec_moire_lattice_gpu = cuda.to_device(rec_moire_lattice.astype(np.float32))
k_points_gpu = cuda.to_device(k_points.astype(np.float32))
c_alpha_gpu = cuda.to_device(c_alpha.astype(np.float32))
R = moire_lattice[3].astype(np.float32)
R_gpu = cuda.to_device(R)

r = bandcalc.generate_monkhorst_pack_set(m, 60)
#size = np.linspace(-5, 5, 40)
#x, y = np.meshgrid(size, size)
#r = np.vstack((x.ravel(), y.ravel())).T
r_gpu = cuda.to_device(r.astype(np.float32))

res_gpu = cuda.device_array((len(r), len(rec_moire_lattice), len(k_points)), dtype=np.complex64)
threadsperblock = (32, 32, 1)
blockspergrid = tuple(res_gpu.shape[i] // threadsperblock[i] if res_gpu.shape[i] % threadsperblock[i] == 0
            else res_gpu.shape[i] // threadsperblock[i] + 1 for i in range(3)
        )

print(threadsperblock)
print(blockspergrid)

@cuda.jit(device=True)
def summand(c_alpha, Q, GM, r, R):
    r"""
    Calculate

    ..math::
        c^{\alpha}_{\mathbf{Q}-\mathbf{G}^{\text{M}}}\text{e}^{-\text{i}
        \mathbf{G}^{\text{M}}\mathbf{r}}\text{e}^{-\text{i}\mathbf{Q}(\mathbf{r}-\mathbf{R}}
    """
    return (c_alpha
            *cmath.exp(
                -1j*(
                    GM[0]*r[0] + GM[1]*r[1]
                    )
                )
            *cmath.exp(
                1j*(
                    Q[0]*(r[0]-R[0]) + Q[1]*(r[1]-R[1])
                    )
                )
            )

@cuda.jit
def wannier(res, c_alpha, Q, GM, r, R):
    """
    Calculate :py:func`sumand` for every needed element
    """
    # i: r
    # j: Q
    # k: GM
    
    i, j, k = cuda.grid(3) #pylint: disable=E0633,E1121

    if i < res.shape[0] and j < res.shape[1] and k < res.shape[2]:
        res[i, j, k] = summand(c_alpha[j, k], Q[j], GM[k], r[i], R)


#@cuda.jit
#def wannier(res, c_alpha, Q, GM, r, R):
#    # i: GM
#    # j: Q
#    i, j = cuda.grid(2)
#    if i < res.shape[1] and j < res.shape[0]:
#        res[j, i] = (c_alpha[j, i]
#            *cmath.exp(
#                -1j*(
#                    GM[i][0]*r[0][0] + GM[i][1]*r[0][1]
#                    )
#                )
#            *cmath.exp(
#                1j*(
#                    Q[j][0]*(r[0][0]-R[0][0]) + Q[j][1]*(r[0][1]-R[0][1])
#                    )
#                )
#            )

## Calculate the wannier function
wannier[blockspergrid, threadsperblock](res_gpu, c_alpha_gpu, k_points_gpu, #pylint: disable=E1136
        rec_moire_lattice_gpu, r_gpu, R_gpu)

res = res_gpu.copy_to_host()
wannier_function = np.sum(res, axis=(1,2))

## Plot the results
pg.mkQApp()
view = gl.GLViewWidget()
view.show()
vertexes = np.concatenate([r, np.abs(wannier_function).reshape(-1, 1)/4000], axis=1)
tri = Delaunay(r)
faces = tri.simplices
mesh_data = gl.MeshData(vertexes=vertexes, faces=faces)
mesh = gl.GLMeshItem(meshdata=mesh_data, color=(0.5, 0.5, 0.5, 1),
        edgeColor=(0.5, 0.5, 0.5, 1), shader="normalColor", smooth=True,
        computeNormals=True)

view.addItem(mesh)
