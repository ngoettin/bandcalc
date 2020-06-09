import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d #pylint: disable=E0611
from mpl_toolkits.mplot3d import Axes3D

import bandcalc

# Constants
a = 1
N = 1000

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
bandstructure = bandcalc.calc_bandstructure(points, lattice, N)

# Plots
fig = plt.figure(figsize=(11,5))
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122)
bandcalc.plot_lattice_3d(ax0, lattice, "ko")
bandcalc.plot_k_path_3d(ax0, path, "r")
bandcalc.plot_bandstructure(ax1, bandstructure, k_names, "k")

ax1.set_ylim([-1, 30])

plt.tight_layout()
plt.show()
