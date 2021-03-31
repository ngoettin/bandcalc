import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d #pylint: disable=E0611
from mpl_toolkits.mplot3d import Axes3D

import bandcalc

# Constants
a = 1
N = 1000
m_e = 0.42*bandcalc.constants.physical_constants["electron mass"][0]
m_h = 0.34*bandcalc.constants.physical_constants["electron mass"][0]
m = m_e+m_h

# Reciprocal lattice vectors
b1 = np.array([2*np.pi/a, 0, 0])
b2 = np.array([0, 2*np.pi/a, 0])
b3 = np.array([0, 0, 2*np.pi/a])
b = np.vstack([b1, b2, b3])
lattice = bandcalc.generate_lattice_by_shell(b, 2)

# k path
points = np.array([[0, 0, 0],    # Gamma
    [0, np.pi/a, 0],             # X
    [np.pi/a, np.pi/a, 0],       # M
    [0, 0, 0],                   # Gamma
    [np.pi/a, np.pi/a, np.pi/a]]) # R
k_names = [r"$\Gamma$", r"X", r"M", r"$\Gamma$", r"R"]
path = bandcalc.generate_k_path(points, N)

# Calculate band structure
potential_matrix = bandcalc.calc_potential_matrix(lattice)
hamiltonian = bandcalc.calc_hamiltonian(lattice, potential_matrix, m)
bandstructure, prefix = bandcalc.get_unit_prefix(
        bandcalc.calc_bandstructure(path, hamiltonian))

# Plots
fig = plt.figure(figsize=(11,5))
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122)
bandcalc.plot_lattice_3d(ax0, lattice, "ko")
bandcalc.plot_k_path_3d(ax0, path, "r")
bandcalc.plot_bandstructure(ax1, bandstructure, k_names, "k")

ax1.set_ylabel(r"$E - \hbar\Omega_0$ in {}eV".format(prefix))
ax1.set_ylim([0, 4])

plt.tight_layout()
plt.show()
