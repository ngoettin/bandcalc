import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d #pylint: disable=E0611

import bandcalc

# Constants
a = 1
N = 1000

# Reciprocal lattice vectors
b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
b = np.vstack([b1, b2])
lattice = bandcalc.generate_lattice_by_shell(b, 1)

# Monkhorst-Pack lattice
mp_lattice = bandcalc.generate_monkhorst_pack_set(b, 10)

# Results
fig, ax = plt.subplots()

vor = Voronoi(lattice)
voronoi_plot_2d(vor, ax, show_points=False, show_vertices=False)
bandcalc.plot_lattice(ax, lattice, "ko", alpha=0.3)
bandcalc.plot_lattice(ax, mp_lattice, "k.")

ax.set_xlim([lattice[:,0].min()-2, lattice[:,0].max()+2])
ax.set_ylim([lattice[:,1].min()-2, lattice[:,1].max()+2])

plt.show()
