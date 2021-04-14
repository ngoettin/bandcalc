import os
import pickle

import ray
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import bandcalc as bc

ray.init(address='auto', _redis_password='5241590000000000')

def U_integrand(r_tilde, r_, R, R_, D, wannier_abs_sqr):
    omega_prime = r_
    omega_tilde = r_tilde - (R - R_ - r_)
    wannier1 = wannier_abs_sqr.ev(omega_tilde[:,0], omega_tilde[:,1])
    wannier2 = wannier_abs_sqr.ev(omega_prime[:,0], omega_prime[:,1])

    length_r_tilde = np.linalg.norm(r_tilde, axis=1)
    try:
        length_r_= np.linalg.norm(r_, axis=1)
    except np.AxisError:
        length_r_= np.linalg.norm(r_)
    potential = length_r_ - length_r_tilde*length_r_/np.sqrt(length_r_tilde**2 + D**2)
    result = wannier1 * wannier2 * potential
    return result

def calc_integral(R_prime, r_max, R0, wannier_spline):
    d2r_, r_ = bc.generate_circle_grid_in_polar(r_max, np.array([0, 0]), num_points, num_points)
    r_tilde_full = []
    d2r_tilde_list = []
    for r_vec in r_:
        d2r_tilde, r_tilde = bc.generate_circle_grid_in_polar(r_max, R0-R_prime-r_vec, num_points, num_points)
        d2r_tilde_list.append(d2r_tilde)
        r_tilde_full.append(r_tilde)
    d2r_tilde = np.array(d2r_tilde_list)
    r_nums = list(map(len, r_tilde_full))
    dV = np.repeat(d2r_*d2r_tilde, r_nums)
    r_tilde_full = np.vstack(r_tilde_full)
    r_full = np.repeat(r_, r_nums, axis=0)
    return np.sum(U_integrand(r_tilde_full, r_full, R0, R_prime, D, wannier_spline)*dV)

D = 3#nm
num_points = 60 # for integration (each dimension)

e = bc.constants.physical_constants["elementary charge"][0]
eps = bc.constants.physical_constants["vacuum electric permittivity"][0]

base_path = os.path.join("bandcalc", "examples" , "misc")
with open(os.path.join(base_path, "wannier_pickle", "moire_vector_lengths.pickle"), "rb") as f:
    m_lens = pickle.load(f)
with open(os.path.join(base_path, "wannier_pickle", "potential_minima.pickle"), "rb") as f:
    potential_minima = pickle.load(f)

@ray.remote
def calc_integrals(angle):
    with open(os.path.join(base_path, "wannier_pickle", f"wannier_{angle:.2f}_deg.pickle"), "rb") as f:
        wannier_spline = pickle.load(f)
    
    angle_str = f"{angle:.2f}"
    print(angle_str+"°")
    m_len = m_lens[angle_str]
    r_max = 2*m_len
    R0, R1, R2 = potential_minima[angle_str]

    U0 = calc_integral(R0, r_max, R0, wannier_spline)
    U1 = calc_integral(R1, r_max, R0, wannier_spline)
    U2 = calc_integral(R2, r_max, R0, wannier_spline)

    return m_len, U0, U1, U2

res = np.array(ray.get([calc_integrals.remote(angle) for angle in np.linspace(1, 3, num=10)]))

fig, ax = plt.subplots()
lines = plt.plot(res[:,0], res[:, 1:]*e/eps*1e9/4/np.pi*1e3)#meV
plt.legend(lines, (r"$U_0$",r"$U_1$",r"$U_2$"))
plt.xlabel(r"$a_M$ in nm")
plt.ylabel(r"$\epsilon_r U$ in meV")
plt.tight_layout()
plt.show()
