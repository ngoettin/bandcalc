import sys
import pickle
import argparse
import configparser

import numpy as np

import bandcalc as bc

parser = argparse.ArgumentParser(description="Calculate the Moire band structure")
parser.add_argument("-s", "--shells", type=int,
        default=1, help="number of lattice shells for the calculation")
parser.add_argument("--config", type=str,
        default="hubbard.conf", help="location of the configuration file")
args = parser.parse_args()
shells = args.shells
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

if V is None or mass is None:
    print("Error: Some constants for this physical system are unknown")
    sys.exit(1)

temperature = config["general"].getfloat("temperature")
filling = config["general"].getfloat("filling")

system_str = f"{material_system.replace('/', '_')}_{stacking}_stacking_{particle_type}"

with open(f"bandcalc/examples/hubbard/wannier_pickle/moire_vector_lengths_{system_str}.pickle", "rb") as f:
    m_lens = pickle.load(f)

res = []
for angle in np.linspace(1, 3, num=10):
    # Reciprocal moire basis vectors
    rec_m = b1-bc.rotate_lattice(b2, angle)

    # Complete reciprocal moire lattice
    rec_moire_lattice = bc.generate_lattice_by_shell(rec_m, shells)

    # Real space moire basis vectors
    m = bc.generate_reciprocal_lattice_basis(rec_m)

    # Real space moire lattice vectors
    moire_lattice = bc.generate_lattice_by_shell(m, 2)

    # Moire potential coefficients
    Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])
    potential_matrix = bc.calc_potential_matrix_from_coeffs(rec_moire_lattice, Vj)

    k_points = bc.generate_monkhorst_pack_set(rec_m, 40)

    ## Calculate the band structure
    # The bandstructure is slightly different from the bandstructure in the
    # original paper, but that is most likely just a small difference in
    # some parameters, like the lattice constants
    hamiltonian = bc.calc_hamiltonian(rec_moire_lattice, potential_matrix, mass)
    bandstructure = np.sort(
            np.real(
                bc.calc_bandstructure(k_points, hamiltonian)
            )
        )*1e3 # in meV

    cell_area = bc.get_volume_element_regular_grid(moire_lattice)
    n = filling/cell_area
    
    mu = bc.calc_mu_of_n_boson(bandstructure, k_points, temperature)(np.log10(n))
    res.append(mu)

    print(f"{angle:.2f}°")

res = np.array(res)
print(res)
np.save(f"mu_{system_str}_{filling}_filling_{temperature}K.npy", res)
