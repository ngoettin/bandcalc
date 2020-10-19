import argparse
import functools

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d #pylint: disable=E0611
from scipy.constants import physical_constants

import bandcalc
from bandcalc.constants import lattice_constants

parser = argparse.ArgumentParser(description="Calculate the Moire band structure")
parser.add_argument("-p", "--potential", choices=["off", "MoS2"],
        default="off", help="choose the potential to calculate the band structure with")
parser.add_argument("-a", "--angle", type=float,
        default=3, help="twist angle of the lattices in degrees")
parser.add_argument("-e", "--energy-level", type=int,
        default=0, help="energy level of the wave function")
parser.add_argument("-s", "--shells", type=int,
        default=1, help="number of lattice shells for the calculation")
args = parser.parse_args()
potential = args.potential
angle = args.angle
energy_level = args.energy_level
shells = args.shells

# Constants
a = lattice_constants["MoS2"]*1e9
N = 1000
m_e = 0.42*bandcalc.constants.physical_constants["electron mass"][0]
m_h = 0.34*bandcalc.constants.physical_constants["electron mass"][0]
m = m_e+m_h

# Reciprocal lattice vectors
b1 = np.array([2*np.pi/(np.sqrt(3)*a), 2*np.pi/a])
b2 = np.array([2*np.pi/(np.sqrt(3)*a), -2*np.pi/a])
b = np.vstack([b1, b2])

# Real space lattice vectors
a = bandcalc.generate_reciprocal_lattice_basis(b)

# Reciprocal moire lattice vectors
rec_m = b-bandcalc.rotate_lattice(b, angle)

# Complete reciprocal moire lattice
rec_moire_lattice = bandcalc.generate_lattice_by_shell(rec_m, shells)

# Find GM
G = bandcalc.generate_twisted_lattice_by_shell(b, b, angle, 1)
GT = G[0]
GB = G[1]
GM = GT-GB
GM = np.array(sorted(GM, key=lambda x: np.abs(x.view(complex))))[1:]
GM = np.array(sorted(GM, key=lambda x: np.angle(x.view(complex))))

# Grid size
size = np.linspace(-50, 50, 500)
grid = np.meshgrid(size, size)

if potential == "MoS2":
    # Moire potential coefficients
    V = 12.4*1e-3*np.exp(1j*81.5*np.pi/180) # in eV
    Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])
    potential_matrix = bandcalc.calc_potential_matrix_from_coeffs(rec_moire_lattice, Vj)
    
    # Moire potential for reference
    moire_potential = bandcalc.calc_moire_potential_on_grid(grid, GM, Vj)

elif potential == "off":
    potential_matrix = bandcalc.calc_potential_matrix(rec_moire_lattice)
    moire_potential = np.zeros(grid[0].shape)

# Calculate MBZ and find a K-point
vor_m = Voronoi(rec_moire_lattice)
sorted_vertices = np.array(sorted(vor_m.vertices, key=lambda x: np.abs(x.view(complex))))
k_point = sorted_vertices[0]

# Calculate the wavefunction
hamiltonian = bandcalc.calc_hamiltonian(rec_moire_lattice, potential_matrix, m)

wavefunction = bandcalc.calc_wave_function_on_grid(k_point, rec_moire_lattice, grid,
        hamiltonian, energy_level)

# Energies for reference
energies, prefix = bandcalc.get_unit_prefix(np.sort(np.linalg.eigvals(
        bandcalc.calc_hamiltonian(rec_moire_lattice, potential_matrix, m)(k_point))))
energy_slice = energies[
    energy_level-5 if energy_level-5>-1 else None:
    energy_level+5 if energy_level+5<len(energies) else None]
energy = np.real(energies[energy_level])

#np.save("wavefunction_moire_{}deg_energy_level_{}_shells_8.npy".format(angle, energy_level), wavefunction)

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

bandcalc.plot_lattice(axs[0], rec_moire_lattice, "ko")
axs[0].text(0.05, 0.95, f"{angle}°", transform=axs[0].transAxes, va="top", bbox={"fc": "white", "alpha": 0.2})
axs[0].set_xlabel(r"nm$^{-1}$")
axs[0].set_ylabel(r"nm$^{-1}$")

bandcalc.plot_moire_potential(axs[1], grid, moire_potential, alpha=0.6, cmap="Greys_r", zorder=0)
bandcalc.plot_wave_function(axs[1], grid, np.abs(wavefunction)**2, alpha=0.4, cmap="jet", zorder=1)
axs[1].set_xlabel("nm")
axs[1].set_ylabel("nm")

ax_ins = axs[1].inset_axes([0.05, 0.05, 0.3, 0.3])
ax_ins.plot(np.vstack([np.zeros(len(energy_slice)), np.ones(len(energy_slice))]),
        np.vstack([energy_slice, energy_slice]), "k")
ax_ins.xaxis.set_visible(False)
ax_ins.yaxis.tick_right()
ax_ins.set_yticks([energy])
ax_ins.set_yticklabels(["{:.2f}{}eV".format(energy, prefix)])
ax_ins.set_xlim([0,1])
ax_ins.patch.set_alpha(0.7)


plt.tight_layout()
plt.show()
