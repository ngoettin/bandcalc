import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import physical_constants
from scipy import interpolate

import bandcalc
from bandcalc.constants import lattice_constants

parser = argparse.ArgumentParser(description="Calculate and pickle spline "\
        "interpolation of wannier function")
parser.add_argument("-p", "--potential", choices=["off", "MoS2"],
        default="off", help="choose the potential to calculate the band structure with")
parser.add_argument("-e", "--energy-level", type=int,
        default=0, help="energy level of the wave function")
parser.add_argument("-s", "--shells", type=int,
        default=1, help="number of lattice shells for the calculation")
parser.add_argument("-o", "--out", type=str,
        default="wannier_pickle", help="where to save the wannier functions")
args = parser.parse_args()
potential = args.potential
energy_level = args.energy_level
shells = args.shells
output_folder = args.out

a = lattice_constants["MoS2"]*1e9
N = 1000
mass = 0.35*physical_constants["electron mass"][0]

# Reciprocal lattice vectors
b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
b = np.vstack([b1, b2])

def make_xy(x_min, x_max, n_x, y_min, y_max, n_y, matrix=False):
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    if matrix:
        x_, y_ = np.meshgrid(x, y)
        xy = np.vstack([x_.ravel(), y_.ravel()]).T
        return xy
    return x, y

m_lens = {}
potential_minima = {}

for angle in np.linspace(1, 3, num=10):
    print(f"{angle}°")
    # Reciprocal moire basis vectors
    rec_m = b-bandcalc.rotate_lattice(b, angle)

    # Complete reciprocal moire lattice
    rec_moire_lattice = bandcalc.generate_lattice_by_shell(rec_m, shells)

    # Real space moire basis vectors
    m = bandcalc.generate_reciprocal_lattice_basis(rec_m)
    m_len = np.abs(m[0].view(complex)[0])

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

    # Wannier functions should be centered around potential minima
    potential_shift = np.array([m[0, 0], m[0, 1]*1/3])
    print(f"Potential shift: {potential_shift}")

    R0 = potential_shift
    R1 = m[0]+potential_shift
    R2 = 2*m[0]+potential_shift

    ## Calculate the wannier function
    x_min, x_max, n_x = -2*m_len, 2*m_len, 100
    y_min, y_max, n_y = -2*m_len, 2*m_len, 100

    x, y = make_xy(x_min, x_max, n_x, y_min, y_max, n_y)
    xy = make_xy(x_min, x_max, n_x, y_min, y_max, n_y, matrix=True)

    wannier_function = bandcalc.calc_wannier_function_gpu(
            hamiltonian, k_points, rec_moire_lattice, xy+R0, R0, band_index=energy_level)

    spline = interpolate.RectBivariateSpline(x, y, np.abs(wannier_function.reshape(n_x, n_y))**2)

    with open(os.path.join(output_folder, f"wannier_{angle:.2f}_deg.pickle"), "wb") as f:
        pickle.dump(spline, f, pickle.HIGHEST_PROTOCOL)

    m_lens[f"{angle:.2f}"] = m_len
    potential_minima[f"{angle:.2f}"] = R0, R1, R2

with open(os.path.join(output_folder, "moire_vector_lengths.pickle"), "wb") as f:
    pickle.dump(m_lens, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(output_folder, "potential_minima.pickle"), "wb") as f:
    pickle.dump(potential_minima, f, pickle.HIGHEST_PROTOCOL)
