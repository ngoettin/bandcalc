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
G = bandcalc.generate_twisted_lattice_by_shell(b, b, angle, 1)
GT = G[0]
GB = G[1]
GM = GT-GB
GM = np.array(sorted(GM, key=lambda x: np.abs(x.view(complex))))[1:]
GM = np.array(sorted(GM, key=lambda x: np.angle(x.view(complex))))

# Real space grid
size = np.linspace(-10, 10, 100)
grid = np.meshgrid(size, size)

# Reciprocal space grid
size_r = np.linspace(-3, 3, 100)
grid_r = np.meshgrid(size_r, size_r)

# Real space moire potential
moire_potential = bandcalc.calc_moire_potential_on_grid(grid, GM, Vj)

# Reciprocal space moire potential
mp_moire = bandcalc.generate_monkhorst_pack_set(m, 100)
moire_potential_pointwise = bandcalc.calc_moire_potential(mp_moire, GM, Vj)
rec_moire_potential = bandcalc.calc_moire_potential_reciprocal_on_grid(
        mp_moire, grid_r, moire_potential_pointwise)

# Results
fig, axs = plt.subplots(nrows=2, ncols=2)

contour = bandcalc.plot_moire_potential(axs[0,0], grid, np.real(moire_potential), alpha=0.4)
moire_lattice = bandcalc.generate_lattice_by_shell(m, 2)
bandcalc.plot_lattice(axs[0,0], moire_lattice, "ko")
vor_m = Voronoi(moire_lattice)
voronoi_plot_2d(vor_m, axs[0,0], show_points=False, show_vertices=False)
bandcalc.plot_lattice(axs[0,0], mp_moire, "b,", alpha=0.4)

rec_moire_lattice = bandcalc.generate_lattice_by_shell(rec_m, 3)

axs[0,0].axis("scaled")
axs[0,0].set_xlim([size.min(), size.max()])
axs[0,0].set_ylim([size.min(), size.max()])

for i, ax in enumerate(axs.ravel()[1:]):
    bandcalc.plot_lattice(ax, rec_moire_lattice, "ko")
    fun = [np.real, np.imag, np.abs][i]
    title = ["real", "imaginary", "absolute"][i]
    rec_contour = bandcalc.plot_moire_potential(ax, grid_r, 
            fun(rec_moire_potential), alpha=0.4)
    ax.axis("scaled")
    ax.set_xlim([size_r.min(), size_r.max()])
    ax.set_ylim([size_r.min(), size_r.max()])
    ax.set_title(title)
    plt.colorbar(rec_contour, ax=ax)

plt.colorbar(contour, ax=axs[0,0], label=r"$\Delta(\mathbf{r})$ in meV")
plt.tight_layout()
plt.show()
