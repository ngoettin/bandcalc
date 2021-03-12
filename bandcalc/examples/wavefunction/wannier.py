import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import physical_constants

import bandcalc
from bandcalc.constants import lattice_constants

parser = argparse.ArgumentParser(description="Calculate wannier function")
parser.add_argument("-p", "--potential", choices=["off", "MoS2"],
        default="off", help="choose the potential to calculate the band structure with")
parser.add_argument("-a", "--angle", type=float,
        default=3, help="twist angle of the lattices in degrees")
parser.add_argument("-e", "--energy-level", type=int,
        default=0, help="energy level of the wave function")
parser.add_argument("-s", "--shells", type=int,
        default=1, help="number of lattice shells for the calculation")
parser.add_argument("--mode", type=str, choices=["2d", "3d"],
    default="3d", help="draw wannier function in 2d or 3d")
args = parser.parse_args()
potential = args.potential
angle = args.angle
energy_level = args.energy_level
shells = args.shells
mode = args.mode

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

if potential == "MoS2":
    # Moire potential coefficients
    V = 6.6*1e-3*np.exp(-1j*94*np.pi/180) # in eV
    Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])
    potential_matrix = bandcalc.calc_potential_matrix_from_coeffs(rec_moire_lattice, Vj)
elif potential == "off":
    potential_matrix = bandcalc.calc_potential_matrix(rec_moire_lattice)

## Calculate eigenstates for every k point in the MBZ
k_points = bandcalc.generate_monkhorst_pack_set(rec_m, 20).astype(np.float32)
hamiltonian = bandcalc.calc_hamiltonian(rec_moire_lattice, potential_matrix, mass)

# Real space moire lattice vectors
moire_lattice = bandcalc.generate_lattice_by_shell(m, shells)

# Wannier functions should be centered around potential minima
potential_shift = np.array([m[0, 0], m[0, 1]*1/3])
print(f"Potential shift: {potential_shift}")

R0 = potential_shift
R1 = m[0]+potential_shift
R2 = 2*m[0]+potential_shift

## Calculate the wannier function
R = R0
r = 4*bandcalc.generate_monkhorst_pack_set(m, 80)#+R

wannier_function = bandcalc.calc_wannier_function_gpu(
        hamiltonian, k_points, rec_moire_lattice, r, R, band_index=energy_level)

## Plot the results
if mode == "3d":
    bandcalc.plot_trisurface_3d(r[:,0], r[:,1], np.abs(wannier_function))
elif mode == "2d":
    fig, ax = plt.subplots()
    contour = ax.tricontourf(r[:,0], r[:,1], np.abs(wannier_function), alpha=0.8)
    bandcalc.plot_lattice(ax, bandcalc.generate_lattice_by_shell(m, 4), "k.")
    ax.plot(R[0], R[1], "ro")
    ax.set_xlabel("nm")
    ax.set_ylabel("nm")
    plt.colorbar(mappable=contour)
    plt.show()
