import numpy as np
import scipy.constants

import ray
ray.init(address='auto', redis_password='5241590000000000')

from .generate import generate_k_path

hbar = scipy.constants.physical_constants["Planck constant over 2 pi"][0]
e = scipy.constants.physical_constants["elementary charge"][0]
m_e = 0.42*scipy.constants.physical_constants["electron mass"][0]
m_h = 0.34*scipy.constants.physical_constants["electron mass"][0]
m = m_e+m_h
V = 0

def eps_0(k, G):
    r"""
    Calculate the unpertubated energy for given k vector and reciprocal lattice vector using

    .. math::
        \varepsilon^{(0)}_{\vec{G}_0}(\vec{k}\,) = \frac{\hbar^2}{2m}\left(\vec{k} - \vec{G}_0\right)^2

    Expects input vectors to be in units 1/nm.

    :param k: k vector of the particle
    :param G: reciprocal lattice vector

    :type k: numpy.ndarray
    :type G: numpy.ndarray

    :rtype: float
    """

    return hbar**2/(2*m)*np.sum((k-G)**2, axis=1)*1e18/e # in eV

def calc_hamiltonian(k, lattice, potential_matrix):
    """
    Construct the hamiltonian for any given k in a specified lattice

    :param k: k vector of the particle
    :param lattice: reciprocal lattice

    :type k: numpy.ndarray
    :type lattice: numpy.ndarray

    :rtype: numpy.ndarray
    """

    diagonal = np.diag(eps_0(k, lattice))
    return potential_matrix + diagonal

def calc_potential_matrix(lattice, potential_fun=None, *args):
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

    if potential_fun is None:
        potential_matrix = np.zeros((lattice.shape[0],)*2)
    elif isinstance(potential_fun, (float, int)):
        potential_matrix = np.ones((lattice.shape[0],)*2)*potential_fun
        np.fill_diagonal(potential_matrix, 0)
    elif isinstance(potential_fun, ray.remote_function.RemoteFunction):
        lattice_matrix = np.array([lattice - vec for vec in lattice])
        potential_matrix = np.array(
                ray.get([potential_fun.remote(lattice, *args) for lattice in lattice_matrix])
        )
    return potential_matrix

def calc_bandstructure(k_points, lattice, N, potential_fun=None, *args):
    """
    Calculate the band structure of a lattice along a given k path with N samples

    :param k_points: k points
    :param lattice: reciprocal lattice
    :param N: number of samples
    :param potential_fun: See :func:`calc_potential_matrix`

    :type k_points: numpy.ndarray
    :type lattice: numpy.ndarray
    :type N: int
    :type potential_fun:

    :rtype: numpy.ndarray
    """

    potential_matrix = calc_potential_matrix(lattice, potential_fun, *args)
    path = generate_k_path(k_points, N)
    eig_vals = np.array(
            [np.linalg.eigvals(calc_hamiltonian(k, lattice, potential_matrix)) for k in path]
    )
    return eig_vals

def calc_wave_function_on_grid(k_point, lattice, grid, energy_level=0, potential_fun=None, *args):
    r"""
    Calculate the wave function (not the absolute square) of a system on a real space grid.
    It is assumed, that the wave function :math:`|\chi\rangle` can be written as

    .. math::
        |\chi_{\mathbf{Q}}\rangle^{(\alpha)}(\mathbf{r}) = \sum_{\mathbf{G}^\text{M}}
        c^{(\alpha)}_{\mathbf{Q}-\mathbf{G}^\text{M}}
        \text{e}^{\text{i}(\mathbf{Q}-\mathbf{G}^\text{M})\mathbf{r}},

    where
     * :math:`\alpha\in \{0, ..., N\}` is the *energy_level* of the wave function.
       If there are :math:`N` reciprocal lattice vectors, there will be :math:`N` energy levels.
     * :math:`\mathbf{G}^{\text{M}}` are the reciprocal *lattice* vectors
     * :math:`\mathbf{r}` are real space vectors (*grid*)
     * :math:`\mathbf{Q}` is the *k_point*, at which the wave function will be evaluated.
     * :math:`c` are the eigenvector solutions of the systems hamiltonian.
       The eigenvectors are sorted by :math:`\alpha` and their components can be referred to
       by the corresponding lattice vectors.

    :param k_point: :math:`\mathbf{Q}`
    :param lattice: :math:`\mathbf{G}^{\text{M}}`
    :param grid: :math:`\mathbf{r}`
    :param energy_level: :math:`\alpha`
    :param potential_fun: See :func:`calc_potential_matrix`

    :type k_point: numpy.ndarray
    :type lattice: numpy.ndarray
    :type grid: list(numpy.ndarray)
    :type energy_level: int
    :type potential_fun:

    :rtype: numpy.ndarray
    """

    if energy_level not in range(len(lattice)):
        raise ValueError(("Energy level {} not allowed. "
                          "There are only {} energy levels in the system "
                          "and counting starts from 0.").format(
                              energy_level, len(lattice)))
    potential_matrix = calc_potential_matrix(lattice, potential_fun, *args)
    hamiltonian = calc_hamiltonian(k_point, lattice, potential_matrix)
    eig_vals, eig_vecs = np.linalg.eig(hamiltonian)

    # Sort the eigenvectors by energy
    eig_vecs = eig_vecs[:, np.argsort(np.real(eig_vals))]

    # Pick the energy level
    eig_vec = eig_vecs[:, energy_level]

    summand = np.exp(1j*(
        np.tensordot((k_point-lattice)[:,0], grid[0], axes=0)+
        np.tensordot((k_point-lattice)[:,1], grid[1], axes=0)
        ))*eig_vec[..., None, None]
    return np.sum(summand, axis=0)

def calc_moire_potential_on_grid(grid, reciprocal_moire_lattice, potential_coeffs):
    r"""
    Calculate the moire potential on a regular grid using

    .. math::
        V^{\text{M}}(\vec{r}) \approx \sum_{j=1}^6 V_j \exp\left(\text{i}\vec{G}_j^{\text{M}}\vec{r}\right)

    :param grid: :math:`\vec{r}`, a numpy meshgrid to calculate the potential on
    :param moire_lattice: :math:`\vec{G}_j^{\text{M}}`, the six moire lattice vectors
    :param potential_coeffs: :math:`V_j`, the coefficients for the potential

    :type grid: list(numpy.ndarray)
    :type moire_lattice: numpy.ndarray
    :type potential_coeffs: numpy.ndarray

    :rtype: numpy.ndarray
    """

    moire_potential = np.sum(np.exp(1j*(
        np.tensordot(reciprocal_moire_lattice[:,0], grid[0], axes=0)+
        np.tensordot(reciprocal_moire_lattice[:,1], grid[1], axes=0)
        ))*potential_coeffs[:, None, None], axis=0)
    return moire_potential

def calc_moire_potential(r, reciprocal_moire_lattice, potential_coeffs):
    r"""
    Calculate the moire potential on a set of points using

    .. math::
        V^{\text{M}}(\vec{r}) \approx \sum_{j=1}^6 V_j \exp\left(\text{i}\vec{G}_j^{\text{M}}\vec{r}\right)

    :param r: :math:`\vec{r}`, set of points
    :param moire_lattice: :math:`\vec{G}_j^{\text{M}}`, the six moire lattice vectors
    :param potential_coeffs: :math:`V_j`, the coefficients for the potential

    :type r: numpy.ndarray
    :type moire_lattice: numpy.ndarray
    :type potential_coeffs: numpy.ndarray

    :rtype: numpy.ndarray
    """

    moire_potential = np.sum(np.exp(1j*(
        np.tensordot(reciprocal_moire_lattice[:,0], r[:,0], axes=0)+
        np.tensordot(reciprocal_moire_lattice[:,1], r[:,1], axes=0)
        ))*potential_coeffs[:, None], axis=0)
    return moire_potential

def calc_moire_potential_reciprocal_on_grid(real_space_points, reciprocal_space_grid, moire_potential_pointwise):
    r"""
    Calculate the reciprocal moire potential on a grid using

    .. math::
        V^{\text{M}}_{G_{\text{M}}} = \frac{1}{A}\int_{\text{MWSC}}
        V_{\text{M}}(\vec{r}\,)\text{e}^{-\text{i}G_{\text{M}}\vec{R}}\text{d}r^2

    with MWSC being the first Moire Wigner Seitz cell.

    :param real_space_points: Real space sample points in the MWSC (for example a Monkhorst-Pack grid)
    :param reciprocal_space_grid: Grid of reciprocal vectors :math:`G_{\text{M}}`
    :param moire_potential_pointwise: Pre-calculated real space Moire potential :math:`V^{\text{M}}(\vec{r}\,)`

    :type real_space_points: numpy.ndarray
    :type reciprocal_space_grid: list(numpy.ndarray)
    :type moire_potential_pointwise: numpy.ndarray

    :rtype: numpy.ndarray
    """

    integrand = np.exp(
            -1j*(
                np.tensordot(real_space_points[:,0], reciprocal_space_grid[0], axes=0) + 
                np.tensordot(real_space_points[:,1], reciprocal_space_grid[1], axes=0)
            ))*moire_potential_pointwise[..., None, None]
    integral = integrand.sum(axis=0)
    return integral/len(real_space_points)

@ray.remote
def calc_moire_potential_reciprocal(reciprocal_space_points, real_space_points, moire_potential_pointwise):
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

    integrand = np.exp(
            -1j*(
                np.tensordot(real_space_points[:,0], reciprocal_space_points[:,0], axes=0) +
                np.tensordot(real_space_points[:,1], reciprocal_space_points[:,1], axes=0)
            ))*moire_potential_pointwise[..., None]
    integral = integrand.sum(axis=0)
    return integral/len(real_space_points)
