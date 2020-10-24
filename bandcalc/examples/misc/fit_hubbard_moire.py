"""
All parameters are from PRL 121 026402 (2018)
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay #pylint: disable=E0611
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

import bandcalc
from bandcalc.constants import lattice_constants, physical_constants

parser = argparse.ArgumentParser(description="Calculate the Moire band structure")
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

## DEBUG
potential = "MoS2"
angle = 2
shells = 2

# Constants
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

## Calculate MBZ and choose some K-points for the k-path
vor_m = Voronoi(rec_moire_lattice)
sorted_vertices = np.array(sorted(vor_m.vertices, key=lambda x: np.abs(x.view(complex))))[:6]
sorted_vertices = np.array(sorted(sorted_vertices, key=lambda x: np.angle(x.view(complex))))
points = np.array([
    [0, 0],
    sorted_vertices[0],
    sorted_vertices[1],
    sorted_vertices[3]])
path = bandcalc.generate_k_path(points, N)
k_names = [r"$\gamma$", r"$\kappa'$", r"$\kappa''$", r"$\kappa$"]

## Calculate the band structure
# The bandstructure is slightly different from the bandstructure in the
# original paper, but that is most likely just a small difference in
# some parameters, like the lattice constants
hamiltonian = bandcalc.calc_hamiltonian(rec_moire_lattice, potential_matrix, mass)
bandstructure, prefix = bandcalc.get_unit_prefix(
        bandcalc.calc_bandstructure(points, N, hamiltonian))

## Fit the Hubbard (Tight Binding) Hamiltonian
# Construct function
number_nearest_neighbours = 3
triangulation = Delaunay(moire_lattice)
def energy_function(k, t1, t2, t3, offset):

    energy_matrix = np.zeros((len(k), len(moire_lattice), len(moire_lattice)), dtype=complex)

    for i, vec in enumerate(moire_lattice):
        for order in range(1, number_nearest_neighbours+1):
            neighbours = bandcalc.find_k_order_delaunay_neighbours(i, triangulation,
                    order, only_k_shell=True, include_point_index=False)
            exp = np.exp(-1j*np.dot(k, (moire_lattice[neighbours]-vec).T))
            t = [t1, t2, t3][order-1]
            energy_matrix[..., i, neighbours] = exp*t
    return np.real(np.sum(energy_matrix, axis=(1,2)) + offset)

# Fit function
lowest_band = np.real(np.sort(bandstructure)[:,0])
popt, pcov = curve_fit(energy_function, path, lowest_band)

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,5))

bandcalc.plot_lattice(axs[0], rec_moire_lattice, "ko")
voronoi_plot_2d(vor_m, axs[0], show_points=False, show_vertices=False)
bandcalc.plot_k_path(axs[0], path, "r")
axs[0].text(0.05, 0.95, f"{angle}°", transform=axs[0].transAxes, va="top", bbox={"fc": "white", "alpha": 0.2})
axs[0].set_xlabel(r"nm$^{-1}$")
axs[0].set_ylabel(r"nm$^{-1}$")

bandcalc.plot_bandstructure(axs[1], np.real(bandstructure), k_names, "k")
axs[1].plot(np.real(np.sort(bandstructure)[:,0]), "k", lw=1)
axs[1].plot(energy_function(path, *popt), "r--")
axs[1].plot([0, len(path)], [0, 0], "k--")
axs[1].set_xlim([0, len(path)])
axs[1].set_ylabel(r"$E - \hbar\Omega_0$ in {}eV".format(prefix))
axs[1].set_ylim([-10, 60])

plt.tight_layout()
plt.show()
