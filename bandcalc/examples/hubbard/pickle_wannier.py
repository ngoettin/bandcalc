import os
import sys
import pickle
import argparse
import configparser

import numpy as np

from scipy import interpolate

import bandcalc as bc

parser = argparse.ArgumentParser(description="Calculate and pickle spline "\
        "interpolation of wannier function")
parser.add_argument("-e", "--energy-level", type=int,
        default=0, help="energy level of the wave function")
parser.add_argument("-s", "--shells", type=int,
        default=1, help="number of lattice shells for the calculation")
parser.add_argument("-o", "--out", type=str,
        default="wannier_pickle", help="where to save the wannier functions")
parser.add_argument("--config", type=str,
        default="hubbard.conf", help="location of the configuration file")
args = parser.parse_args()
energy_level = args.energy_level
shells = args.shells
output_folder = args.out
config_file = args.config

config = configparser.ConfigParser()
config.read(config_file)

materials = config["general"]["material_system"].split("/")
a1 = bc.constants.lattice_constants[materials[0]]
a2 = bc.constants.lattice_constants[materials[1]]

b1 = np.array([[2*np.pi/(np.sqrt(3)*a1), 2*np.pi/a1],
    [2*np.pi/(np.sqrt(3)*a1), -2*np.pi/a1]])
b2 = np.array([[2*np.pi/(np.sqrt(3)*a2), 2*np.pi/a2],
    [2*np.pi/(np.sqrt(3)*a2), -2*np.pi/a2]])

particle_type = config["general"]["particle_type"]
material_system = config["general"]["material_system"]
stacking = config["general"]["stacking"]
V = bc.constants.V.get(particle_type, {}).get(material_system, {}).get(stacking)
mass = bc.constants.mass.get(particle_type, {}).get(material_system, {}).get(stacking)

system_str = f"{material_system.replace('/', '_')}_{stacking}_stacking_{particle_type}"

if V is None or mass is None:
    print("Error: Some constants for this physical system are unknown")
    sys.exit(1)

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
    rec_m = b1-bc.rotate_lattice(b2, angle)

    # Complete reciprocal moire lattice
    rec_moire_lattice = bc.generate_lattice_by_shell(rec_m, shells)

    # Real space moire basis vectors
    m = bc.generate_reciprocal_lattice_basis(rec_m)
    m_len = np.abs(m[0].view(complex)[0])

    # Moire potential coefficients
    Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])
    potential_matrix = bc.calc_potential_matrix_from_coeffs(rec_moire_lattice, Vj)

    ## Calculate eigenstates for every k point in the MBZ
    k_points = bc.generate_monkhorst_pack_set(rec_m, 30).astype(np.float32)
    hamiltonian = bc.calc_hamiltonian(rec_moire_lattice, potential_matrix, mass)

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

    wannier_function = bc.calc_wannier_function_gpu(
            hamiltonian, k_points, rec_moire_lattice, xy+R0, R0, band_index=energy_level)

    spline = interpolate.RectBivariateSpline(x, y, np.abs(wannier_function.reshape(n_x, n_y))**2)

    with open(os.path.join(output_folder, f"wannier_{system_str}_{angle:.2f}_deg.pickle"), "wb") as f:
        pickle.dump(spline, f, pickle.HIGHEST_PROTOCOL)

    m_lens[f"{angle:.2f}"] = m_len
    potential_minima[f"{angle:.2f}"] = R0, R1, R2

with open(os.path.join(output_folder, f"moire_vector_lengths_{system_str}.pickle"), "wb") as f:
    pickle.dump(m_lens, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(output_folder, f"potential_minima_{system_str}.pickle"), "wb") as f:
    pickle.dump(potential_minima, f, pickle.HIGHEST_PROTOCOL)
