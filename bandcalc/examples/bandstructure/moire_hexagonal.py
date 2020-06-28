import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d #pylint: disable=E0611
from scipy.constants import physical_constants

import bandcalc
from bandcalc.constants import lattice_constants

# Constants
a = lattice_constants["MoS2"]*1e9
N = 1000
angle = 3

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
GT = G[0,1:]
GB = G[1,1:]
GM = GT-GB

# Sort the reciprocal moire vectors by angle to get the phase right
GM = np.array(sorted(GM, key=lambda x: np.angle(x.view(complex))))

# Complete reciprocal moire lattice
rec_moire_lattice = bandcalc.generate_lattice_by_shell(rec_m, 1)

# Calculate MBZ and choose some K-points for the k-path
vor_m = Voronoi(rec_moire_lattice)
points = np.array([
    vor_m.vertices[0],
    [0, 0],
    vor_m.vertices[1]])
path = bandcalc.generate_k_path(points, N)
k_names = [r"$\kappa$", r"$\gamma$", r"$\kappa$"]

# Calculate the band structure (no potential)
bandstructure = (bandcalc.calc_bandstructure(points, rec_moire_lattice, N) # nm^2kg/s
        * 1e-18 # convert to m^2kg/2
        / physical_constants["elementary charge"][0] # convert to eV
        * 1e3 # convert to meV
)

# Plot the results

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,5))

bandcalc.plot_lattice(axs[0], rec_moire_lattice, "ko")
voronoi_plot_2d(vor_m, axs[0], show_points=False, show_vertices=False)
bandcalc.plot_k_path(axs[0], path, "r")
axs[0].text(0.05, 0.95, f"{angle}°", transform=axs[0].transAxes, va="top", bbox={"fc": "white", "alpha": 0.2})
axs[0].set_xlabel(r"nm")
axs[0].set_ylabel(r"nm")

bandcalc.plot_bandstructure(axs[1], bandstructure, k_names, "k")
axs[1].set_ylabel(r"$E - \hbar\Omega_0$ in meV")

plt.tight_layout()
plt.show()
