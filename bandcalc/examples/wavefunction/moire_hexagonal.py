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

#pool = Pool(processes=4)

# Constants
a = lattice_constants["MoS2"]*1e9
N = 1000

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

if potential == "MoS2":
    # Moire potential coefficients
    V = 12.4*1e-3*np.exp(1j*81.5*np.pi/180) # in eV
    Vj = np.array([V if i%2 else np.conjugate(V) for i in range(1, 7)])
    # Reciprocal moire lattice vectors
    G = bandcalc.generate_twisted_lattice_by_shell(b, b, angle, 1)
    GT = G[0,1:]
    GB = G[1,1:]
    GM = GT-GB

    # Sort the reciprocal moire vectors by angle to get the phase right
    GM = np.array(sorted(GM, key=lambda x: np.angle(x.view(complex))))

    # Generate a real space monkhorst pack lattice
    mp_moire = bandcalc.generate_monkhorst_pack_raw(m, 100)

    # Calculate pointwise real space moire potential
    moire_potential_pointwise = bandcalc.calc_moire_potential(mp_moire, GM, Vj)

    #potential_fun = functools.partial(bandcalc.calc_moire_potential_reciprocal,
    #        real_space_points=mp_moire, moire_potential_pointwise=moire_potential_pointwise)
    potential_fun = bandcalc.calc_moire_potential_reciprocal
elif potential == "off":
    potential_fun = None

# Complete reciprocal moire lattice
rec_moire_lattice = bandcalc.generate_lattice_by_shell(rec_m, shells)

# Calculate MBZ and find a K-point
vor_m = Voronoi(rec_moire_lattice)
sorted_vertices = np.array(sorted(vor_m.vertices, key=lambda x: np.abs(x.view(complex))))
k_point = sorted_vertices[0]
#k_point = np.array([0, 0])

# Calculate the wavefunction
size = np.linspace(-50, 50, 500)
grid = np.meshgrid(size, size)

wavefunction = bandcalc.calc_wave_function_on_grid(k_point, rec_moire_lattice, grid,
        energy_level, potential_fun, mp_moire, moire_potential_pointwise)

# Moire potential for reference
moire_potential = bandcalc.calc_moire_potential_on_grid(grid, GM, Vj)

# Energies for reference
potential_matrix = bandcalc.calc_potential_matrix(rec_moire_lattice, potential_fun, mp_moire, moire_potential_pointwise)
energies, prefix = bandcalc.get_unit_prefix(np.sort(np.linalg.eigvals(
        bandcalc.calc_hamiltonian(k_point, rec_moire_lattice, potential_matrix))))
energy_slice = energies[
    energy_level-5 if energy_level-5>-1 else None:
    energy_level+5 if energy_level+5<len(energies) else None]
energy = np.real(energies[energy_level])

#np.save("wavefunction_moire_{}deg_energy_level_{}_shells_8.npy".format(angle, energy_level), wavefunction)

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

bandcalc.plot_lattice(axs[0], rec_moire_lattice, "ko")
    
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
