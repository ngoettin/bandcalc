import time
import json

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import ray
ray.init(address='auto', redis_password='5241590000000000', ignore_reinit_error=True)

import bandcalc
from bandcalc.constants import lattice_constants

#############################
######### Functions #########
#############################

def calc_potential_matrix(lattice, use_gpu=False, potential_fun=None, *args):
    """
    Calculate matrix of potentials using *potential_fun*.

    :param lattice: reciprocal lattice
    :param potential_fun: function that calculates potential for a set of lattice vectors.
                          Has to be a `ray.remote_function.RemoteFunction` or an `int`/`float`
                          for a constant potential

    :type lattice: numpy.ndarray
    :type potential_fun: ray.remote_function.RemoteFunction | int | float

    :rtype: numpy.ndarray
    """

    if use_gpu:
        xp = cp
        lattice_matrix = xp.array([lattice - vec for vec in lattice])
        potential_matrix = xp.array(
                ray.get([potential_fun.remote(lattice, *args) for lattice in lattice_matrix])
        )
    else:
        xp = np
        lattice_matrix = xp.array([lattice - vec for vec in lattice])
        potential_matrix = xp.array(
                ray.get([potential_fun.remote(lattice, *args) for lattice in lattice_matrix])
        )

    return potential_matrix

def calc_moire_potential_reciprocal(reciprocal_space_points, real_space_points, moire_potential_pointwise,
        use_gpu=False):
    r"""
    Calculate the reciprocal moire potential using

    .. math::
        V^{\text{M}}_{G_{\text{M}}} = \frac{1}{A}\int_{\text{MWSC}}
        V_{\text{M}}(\vec{r}\,)\text{e}^{-\text{i}G_{\text{M}}\vec{R}}\text{d}r^2

    with MWSC being the first Moire Wigner Seitz cell.

    :param real_space_points: Real space sample points in the MWSC (for example a Monkhorst-Pack grid)
    :param reciprocal_space_grid: Reciprocal vectors :math:`G_{\text{M}}`
    :param moire_potential_pointwise: Pre-calculated real space Moire potential :math:`V^{\text{M}}(\vec{r}\,)`

    :type real_space_points: numpy.ndarray
    :type reciprocal_space_grid: numpy.ndarray
    :type moire_potential_pointwise: numpy.ndarray

    :rtype: numpy.ndarray
    """

    xp = cp if use_gpu else np

    integrand = xp.exp(
            -1j*(
                xp.tensordot(real_space_points[:,0], reciprocal_space_points[:,0], axes=0) +
                xp.tensordot(real_space_points[:,1], reciprocal_space_points[:,1], axes=0)
            ))*moire_potential_pointwise[..., None]
    integral = integrand.sum(axis=0)
    return integral/len(real_space_points)

#@ray.remote(num_gpus=1)
#def calc_moire_potential_reciprocal_gpu(reciprocal_space_points, real_space_points, moire_potential_pointwise):
#    r"""
#    Calculate the reciprocal moire potential using
#
#    .. math::
#        V^{\text{M}}_{G_{\text{M}}} = \frac{1}{A}\int_{\text{MWSC}}
#        V_{\text{M}}(\vec{r}\,)\text{e}^{-\text{i}G_{\text{M}}\vec{R}}\text{d}r^2
#
#    with MWSC being the first Moire Wigner Seitz cell.
#
#    :param real_space_points: Real space sample points in the MWSC (for example a Monkhorst-Pack grid)
#    :param reciprocal_space_grid: Reciprocal vectors :math:`G_{\text{M}}`
#    :param moire_potential_pointwise: Pre-calculated real space Moire potential :math:`V^{\text{M}}(\vec{r}\,)`
#
#    :type real_space_points: numpy.ndarray
#    :type reciprocal_space_grid: numpy.ndarray
#    :type moire_potential_pointwise: numpy.ndarray
#
#    :rtype: numpy.ndarray
#    """
#
#    xp = cp
#
#    integrand = xp.exp(
#            -1j*(
#                xp.tensordot(real_space_points[:,0], reciprocal_space_points[:,0], axes=0) +
#                xp.tensordot(real_space_points[:,1], reciprocal_space_points[:,1], axes=0)
#            ))*moire_potential_pointwise[..., None]
#    integral = integrand.sum(axis=0)
#    return integral/len(real_space_points)

#############################
####### Variables ###########
#############################

# Constants
a = lattice_constants["MoS2"]*1e9
angle = 1

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

moire_potential_pointwise = bandcalc.calc_moire_potential(mp_moire, GM, Vj)

results = {"CPU": [],
        "GPU": []
        }

########################
###### Comparison ######
########################

shells = [1, 2]#, 4, 8, 10, 12, 14, 16]
for shell in shells:
    print(f"Shells: {shell}")
    # Complete reciprocal moire lattice
    rec_moire_lattice = bandcalc.generate_lattice_by_shell(rec_m, shell)
    size = len(rec_moire_lattice)

    #### CPU ####

    cpu_fun = ray.remote(calc_moire_potential_reciprocal)

    print("CPU calculating...")
    start = time.time()
    calc_potential_matrix(rec_moire_lattice, False,
            cpu_fun, mp_moire, moire_potential_pointwise, False)
    results["CPU"].append([size, time.time()-start])
    print("Done.")

    #### GPU ####

    rec_moire_lattice_gpu = cp.asarray(rec_moire_lattice)
    mp_moire_gpu = cp.asarray(mp_moire)
    moire_potential_pointwise_gpu = cp.asarray(moire_potential_pointwise)

    gpu_fun = ray.remote(num_gpus=1)(calc_moire_potential_reciprocal)
    
    print("GPU calculating...")
    start = time.time()
    calc_potential_matrix(rec_moire_lattice_gpu, True,
            gpu_fun, mp_moire_gpu, moire_potential_pointwise_gpu, True)
    results["GPU"].append([size, time.time()-start])
    print("Done.")

cpu = np.array(results["CPU"])
gpu = np.array(results["GPU"])

plt.plot(cpu[:, 0], cpu[:, 1], "b")
plt.plot(gpu[:, 0], gpu[:, 1], "g")
plt.figure()
plt.bar(cpu[:, 0], cpu[:, 1]/gpu[:, 1])
plt.xscale("log")
plt.show()

with open("results.json", "w") as f:
    json.dump(results, f)
