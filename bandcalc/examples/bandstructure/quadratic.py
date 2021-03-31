import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d #pylint: disable=E0611

import bandcalc

# Constants
a = 1
N = 1000
m_e = 0.42*bandcalc.constants.physical_constants["electron mass"][0]
m_h = 0.34*bandcalc.constants.physical_constants["electron mass"][0]
m = m_e+m_h

# Reciprocal lattice vectors
b1 = np.array([2*np.pi/a, 0])
b2 = np.array([0, 2*np.pi/a])
b = np.vstack([b1, b2])
lattice = bandcalc.generate_lattice_by_shell(b, 2)

# k path
points = np.array([[0, 0],  # Gamma
    [np.pi/a, 0],           # X
    [np.pi/a, np.pi/a],     # M
    [0, 0]])                # Gamma
k_names = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]
path = bandcalc.generate_k_path(points, N)

# Calculate Brillouin zones
vor = Voronoi(lattice)

# Calculate band structure
potential_matrix = bandcalc.calc_potential_matrix(lattice)
hamiltonian = bandcalc.calc_hamiltonian(lattice, potential_matrix, m)
bandstructure, prefix = bandcalc.get_unit_prefix(
        bandcalc.calc_bandstructure(path, hamiltonian))

# Plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
bandcalc.plot_lattice(ax[0], lattice, "ko")
bandcalc.plot_k_path(ax[0], path, "r")
voronoi_plot_2d(vor, ax[0], show_points=False, show_vertices=False)
bandcalc.plot_bandstructure(ax[1], bandstructure, k_names, "k")

ax[1].set_ylabel(r"$E - \hbar\Omega_0$ in {}eV".format(prefix))
ax[1].set_ylim([0, 4])

plt.tight_layout()
plt.show()
