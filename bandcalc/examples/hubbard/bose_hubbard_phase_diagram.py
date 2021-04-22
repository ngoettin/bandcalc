import sys
import argparse
import configparser

import numba as nb
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Calculate the Bose-Hubbard phase "\
        "diagram and plot the curve of the reachable parameter set in it.")
parser.add_argument("--config", type=str,
        default="hubbard.conf", help="location of the configuration file")
args = parser.parse_args()
config_file = args.config

config = configparser.ConfigParser()
config.read(config_file)

particle_type = config["general"]["particle_type"]
material_system = config["general"]["material_system"]
stacking = config["general"]["stacking"]
filling = config["general"].getfloat("filling")
temperature = config["general"].getfloat("temperature")
eps_r = config["general"].getfloat("eps_r")

system_str = f"{material_system.replace('/', '_')}_{stacking}_stacking_{particle_type}"

if particle_type != "exciton":
    print(f"Error: Particle type {particle_type} is not bosonic")
    sys.exit(1)

U0 = np.load(f"U0_meV_{system_str}.npy")/eps_r
t1 = np.load(f"t1_meV_{system_str}.npy")
mu = np.load(f"mu_meV_{system_str}_{filling}_filling_{temperature}K.npy")

# maximum number of bosons on a lattice site
n_max = 15
# system dimension
d = 2
# lattice coordination number
z = 2*d

steps_mu = 500
steps_J = 500
mu_range = np.linspace(0, np.abs(mu/U0).max()*1.1, steps_mu)
J_range = np.linspace(0, np.abs(t1/U0).max()*1.1, steps_J)

q = np.arange(n_max)
b = np.diag(np.sqrt(q[1:]), k=1)
b_dagger = b.T
n = b_dagger @ b

@nb.njit
def hamiltonian(phi, mu, J):
    U = 1 
    main_diag = np.diag(-z*J*np.abs(phi)**2 + 0.5*U*q*(q-1) - mu*q)
    upper_diag = np.diag(-z*J*phi*np.sqrt(q[1:]), k=1)
    lower_diag = np.diag(-z*J*np.conj(phi)*np.sqrt(q[1:]), k=-1)
    operator = main_diag + upper_diag + lower_diag
    return operator

@nb.njit(parallel=True)
def calc_phi():
    res = np.zeros((2, steps_mu, steps_J))
    count = 0
    for i in nb.prange(steps_mu):
        mu = mu_range[i]
        for j in range(steps_J):
            J = J_range[j]
            phi = 1
            phi_old = np.inf
            while np.abs(phi-phi_old)>1e-4:
                H = hamiltonian(phi, mu, J)
                w, v = np.linalg.eig(H)
                ground_state = v[:,np.argmin(w)]

                phi_old = phi
                phi = ground_state @ b @ ground_state
            res[0,i,j] = phi
            res[1,i,j] = ground_state @ n @ ground_state
        if not count%10:
            print("ca.", np.round(count/steps_mu*4*100, 2), "%")
        count += 1
    return res

def calc_phase_border_J(m, mu):
    U=1
    return -1/z/(m/(U*(m-1)-mu) + (m+1)/(mu-U*m))

res = calc_phi()
plt.imshow(res[0], extent=[J_range.min(), J_range.max(),
    mu_range.min(), mu_range.max()], aspect="auto",
    origin="lower")

plt.plot(np.abs(t1/U0), np.abs(mu/U0), "r")

plt.xlim([J_range.min(), J_range.max()])
plt.ylim([mu_range.min(), mu_range.max()])
plt.xlabel(r"$J/U$")
plt.ylabel(r"$\mu/U$")
plt.colorbar(label=r"$\phi$")

plt.figure()
plt.imshow(res[1], extent=[J_range.min(), J_range.max(),
    mu_range.min(), mu_range.max()], aspect="auto",
    origin="lower", cmap="turbo")
plt.xlim([J_range.min(), J_range.max()])
plt.ylim([mu_range.min(), mu_range.max()])
plt.xlabel(r"$J/U$")
plt.ylabel(r"$\mu/U$")
plt.colorbar(label=r"$n$")
plt.show()
