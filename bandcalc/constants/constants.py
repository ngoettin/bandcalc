import cmath
from scipy.constants import physical_constants

unit_prefixes = {
        -9: "n",
        -6: r"$\mu$",
        -3: "m",
        0:  "",
        3:  "k",
        6:  "M",
        9:  "G",
}

# in nm
layer_distance = {
        "exciton": { # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.97.035306
            # +0.3nm is an arbitrary estimation for layer thickness and has to be verified
            "MoS2/WS2": 0.615 + 0.3,
            "MoSe2/WSe2": 0.647 + 0.3,
        },
        "electron": { # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.026402
            "MoSe2/WSe2": 3,
        }
}

# in eV
V = {
    "exciton" : {
        "MoS2/WS2": { # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.97.035306
            "AA": 12.4*1e-3*cmath.exp(1j*81.5*cmath.pi/180),
            "AB": 1.8*1e-3*cmath.exp(1j*154.5*cmath.pi/180),
        },
    },
    "electron": {
        "MoSe2/WSe2": { # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.026402
            "AA": 6.6*1e-3*cmath.exp(-1j*94*cmath.pi/180),
        },
        "MoS2/WSe2": { # https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.121.026402/Moire_Hubbard_SM.pdf
            "AA": 5.1*1e-3*cmath.exp(-1j*71*cmath.pi/180),
        },
    }
}

# in kg
mass = {
        "exciton": { # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.97.035306
            "MoS2/WS2": {
                "AA": (0.42 + 0.34) * physical_constants["electron mass"][0],
                "AB": (0.42 + 0.34) * physical_constants["electron mass"][0],
            },
            "MoSe2/WSe2": {
                "AA": (0.49 + 0.35) * physical_constants["electron mass"][0],
                "AB": (0.49 + 0.35) * physical_constants["electron mass"][0],
            },
        },
        "electron": { # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.026402
            "MoSe2/WSe2": {
                "AA": 0.35 * physical_constants["electron mass"][0],
                "AB": 0.35 * physical_constants["electron mass"][0],
            },
        }
}

# in nm
lattice_constants = {
        "MoS2": 0.319, # https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.121.026402/Moire_Hubbard_SM.pdf
        "MoSe2": 0.329, # https://www.nature.com/articles/nmat4064.pdf
        "WS2": 0.319, # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.97.035306
        "WSe2": 0.332, # https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.121.026402/Moire_Hubbard_SM.pdf
}
