import sys
import argparse
import configparser

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from matplotlib.patches import Patch

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
angles = np.linspace(1, 3, num=10)

# mu range for interpolation (need many points, because curve gets
# very steep for low mu
mu_inter = np.linspace(np.abs(mu/U0).min(), np.abs(mu/U0).max(), num=100000)
angles_inter = scipy.interpolate.interp1d(np.abs(mu/U0), angles, kind="cubic")(mu_inter)
# maximum number of bosons on a lattice site
n_max = 15
# system dimension
d = 2
# lattice coordination number
z = 2*d

def calc_phase_border_J(m, mu):
    U=1
    return -1/z/(m/(U*(m-1)-mu) + (m+1)/(mu-U*m))

# Calculate phase border
mu_border = []
J_border = []
for m in range(1, int(np.ceil(np.abs(mu/U0).max()*1.1)+1)):
    mu_space = np.linspace(m-1, m)
    mu_border.append(mu_space)
    J_border.append(calc_phase_border_J(m, mu_space))
# There may be duplicates in the border
mu_border, indices = np.unique(np.hstack(mu_border), return_index=True)
J_border = np.hstack(J_border)[indices]
J_border_inter = scipy.interpolate.interp1d(mu_border, J_border, kind="cubic")(mu_inter)

# Calculate interpolation of reachable parameters
J_inter = scipy.interpolate.interp1d(np.abs(mu/U0), np.abs(t1/U0), kind="cubic")(mu_inter)

# Calculate intersections
intersection_indices = np.argwhere(np.diff(np.sign(J_border_inter - J_inter))).flatten()

# Plot
plt.plot(J_border_inter, mu_inter)
plt.plot(J_inter, mu_inter)
plt.plot(J_inter[intersection_indices], mu_inter[intersection_indices], "ko")

plt.figure()
angles = np.hstack([3, angles_inter[intersection_indices], 1])
is_suprafluid = J_inter[0]>J_border_inter[0]
colors = plt.get_cmap("viridis")([0.25, 0.75])
for i in range(len(angles)-1):
    curr_angle = angles[i]
    next_angle = angles[i+1]
    plt.barh(0, next_angle-curr_angle, left=curr_angle, height=1,
            color=(colors[0] if is_suprafluid else colors[1]))
    is_suprafluid = not is_suprafluid
plt.gca().yaxis.set_visible(False)
plt.xlim([1, 3])
plt.ylim([-0.5, 0.5])
plt.xlabel(r"twist angle $\theta$ in deg")
plt.legend(
    [Patch(color=colors[0]), Patch(color=colors[1])],
    ["Supra fluid", "Mott isolator"]
)
plt.show()
