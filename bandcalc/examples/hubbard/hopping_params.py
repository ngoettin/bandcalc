import sys
import pickle
import argparse
import configparser

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay #pylint: disable=E0611
from scipy.optimize import curve_fit

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
    moire_lattice = bc.generate_lattice_by_shell(m, shells)

    # Moire potential coefficients
    Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])
    potential_matrix = bc.calc_potential_matrix_from_coeffs(rec_moire_lattice, Vj)

    ## Calculate MBZ and choose some K-points for the k-path
    vor_m = Voronoi(rec_moire_lattice)
    sorted_vertices = np.array(sorted(vor_m.vertices, key=lambda x: np.abs(x.view(complex))))[:6]
    sorted_vertices = np.array(sorted(sorted_vertices, key=lambda x: np.angle(x.view(complex))))
    points = np.array([
        [0, 0],
        sorted_vertices[0],
        sorted_vertices[1],
        sorted_vertices[3]])
    path = bc.generate_k_path(points, 1000)
    k_names = [r"$\gamma$", r"$\kappa'$", r"$\kappa''$", r"$\kappa$"]

    ## Calculate the band structure
    # The bandstructure is slightly different from the bandstructure in the
    # original paper, but that is most likely just a small difference in
    # some parameters, like the lattice constants
    hamiltonian = bc.calc_hamiltonian(rec_moire_lattice, potential_matrix, mass)
    bandstructure = bc.calc_bandstructure(path, hamiltonian)

    ## Fit the Hubbard (Tight Binding) Hamiltonian
    # Construct function
    number_nearest_neighbours = 2
    triangulation = Delaunay(moire_lattice)
    zero_vec_ind = bc.find_vector_index(moire_lattice, [0, 0])
    nn = np.vstack([moire_lattice[bc.find_k_order_delaunay_neighbours(zero_vec_ind, triangulation, k,
           only_k_shell=True, include_point_index=False)] for k in range(1, number_nearest_neighbours+1)])
    t_lengths = [len(bc.find_k_order_delaunay_neighbours(zero_vec_ind, triangulation, k,
           only_k_shell=True, include_point_index=False)) for k in range(1, number_nearest_neighbours+1)]

    def energy_function(k, t1, t2, offset):
       t = np.hstack([[t1]*t_lengths[0], [t2]*t_lengths[1]])
       energy = np.real(np.dot(t, np.exp(-1j*np.dot(k, nn.T).T)))
       return energy + offset

    # Fit function
    lowest_band = np.real(np.sort(bandstructure)[:,0])
    popt, pcov = curve_fit(energy_function, path, lowest_band)

    with open("res.txt", "a+") as f:
        f.write(str(angle) + " " + " ".join(popt.astype(str)) + "\n")

    res.append(np.hstack([angle, *popt]))

    print(f"{angle:.2f}°")

res = np.array(res)
np.save(f"t1_meV_{system_str}.npy", res[:,1]*1e3)

plt.plot(list(m_lens.values()), res[:,1:3]*1e3)
plt.legend([rf"$t_{i}$" for i in [1,2,3]])
plt.xlabel("$a_M$ in nm")
plt.ylabel("t in meV")
plt.tight_layout()
plt.show()
