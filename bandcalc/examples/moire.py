import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d #pylint: disable=E0611

import bandcalc

# Constants
a = 1
N = 1000
angle = 7.34

# Reciprocal lattice vectors
b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
b = np.vstack([b1, b2])

# Real space lattice vectors
a = bandcalc.generate_reciprocal_lattice_basis(b)

# Reciprocal moire lattice vectors
rec_m = b-bandcalc.rotate_lattice(b, angle)

# Real space moire lattice vectors
m = bandcalc.generate_reciprocal_lattice_basis(rec_m)

# Moire potential coefficients
V = 12.4*np.exp(1j*81.5*np.pi/180)
Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])

# Reciprocal moire lattice vectors
G = bandcalc.generate_moire_lattice_by_shell(b, b, angle, 1)
GT = G[0,1:]
GB = G[1,1:]
GM = GT-GB

# Real space grid
size = np.linspace(-10, 10, 100)
grid = np.meshgrid(size, size)

# Results
fig, ax = plt.subplots()

moire_potential = bandcalc.calc_moire_potential(grid, GM, Vj)
bandcalc.plot_moire_potential(ax, grid, moire_potential, alpha=0.4)

twisted_lattice = bandcalc.generate_moire_lattice_by_shell(a, a, angle, 15)
bandcalc.plot_lattice(ax, twisted_lattice[0], "r.")
bandcalc.plot_lattice(ax, twisted_lattice[1], "b.")

moire_lattice = bandcalc.generate_lattice_by_shell(m, 1)
bandcalc.plot_lattice(ax, moire_lattice, "ko")

ax.axis("scaled")
ax.set_xlim([size.min(), size.max()])
ax.set_ylim([size.min(), size.max()])
plt.show()
