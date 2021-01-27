import argparse
import threading

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import physical_constants
from scipy.spatial import Voronoi #pylint: disable=E0611

import bandcalc
from bandcalc.constants import lattice_constants

parser = argparse.ArgumentParser(description="Calculate wannier function")
parser.add_argument("-p", "--potential", choices=["off", "MoS2"],
        default="off", help="choose the potential to calculate the band structure with")
parser.add_argument("-a", "--angle", type=float,
        default=3, help="twist angle of the lattices in degrees")
parser.add_argument("-s", "--shells", type=int,
        default=1, help="number of lattice shells for the calculation")
args = parser.parse_args()
potential = args.potential
angle = args.angle
shells = args.shells

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
k_points = bandcalc.generate_monkhorst_pack_set(rec_m, 30).astype(np.float32)
hamiltonian = bandcalc.calc_hamiltonian(rec_moire_lattice, potential_matrix, mass)

## Calculate MBZ and choose some K-points for the k-path
vor_m = Voronoi(rec_moire_lattice)
sorted_vertices = np.array(sorted(vor_m.vertices, key=lambda x: np.abs(x.view(complex))))[:6]
sorted_vertices = np.array(sorted(sorted_vertices, key=lambda x: np.angle(x.view(complex))))
points = np.array([
    [0, 0],
    sorted_vertices[0],
    sorted_vertices[1],
    sorted_vertices[3]])
k_names = [r"$\gamma$", r"$\kappa'$", r"$\kappa''$", r"$\kappa$"]

## Calculate bandstructure
bandstructure, prefix = bandcalc.get_unit_prefix(
        bandcalc.calc_bandstructure(points, N, hamiltonian))

sorted_bandstructure = np.sort(bandstructure)

fig, ax = plt.subplots(nrows=7, ncols=2, figsize=(7, 20))
for i in range(7):
    ## Calculate the wannier function
    R = 1/3*(moire_lattice[5]+moire_lattice[3]+moire_lattice[6]).astype(np.float32)
    r = 5*bandcalc.generate_monkhorst_pack_set(m, 80)

    wannier_function = bandcalc.calc_wannier_function_gpu(
            hamiltonian, k_points, rec_moire_lattice, r, R, band_index=i)

    ## Plot the results
    bandcalc.plot_bandstructure(ax[i, 0], np.real(bandstructure), k_names, "k", alpha=0.3)
    ax[i, 0].plot(np.real(sorted_bandstructure)[:, i], "k", lw=2)

    ax[i, 1].tricontourf(r[:,0], r[:,1], np.abs(wannier_function), alpha=0.8)
    lat = bandcalc.generate_lattice_by_shell(m, 3)
    bandcalc.plot_lattice(ax[i, 1], lat, "ko")
    ax[i, 1].plot(R[0], R[1], "ro")
    for num, vec in enumerate(moire_lattice):
        ax[i, 1].text(vec[0]+0.2, vec[1], num, color="#999")

plt.tight_layout()
plt.savefig("wannier_all_bands.pdf")

