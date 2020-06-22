import numpy as np

from .generate import generate_k_path

hbar = 1
m = 1
V = 0

def eps_0(k, G):
    r"""
    Calculate the unpertubated energy for given k vector and reciprocal lattice vector using

    .. math::
        \varepsilon^{(0)}_{\vec{G}_0}(\vec{k}\,) = \frac{\hbar^2}{2m}\left(\vec{k} - \vec{G}_0\right)^2

    :param k: k vector of the particle
    :param G: reciprocal lattice vector

    :type k: numpy.ndarray
    :type G: numpy.ndarray

    :rtype: float
    """

    return hbar**2/(2*m)*np.sum((k-G)**2, axis=1)

def matrix(k, lattice):
    """
    Construct the hamiltonian for any given k in a specified lattice

    :param k: k vector of the particle
    :param lattice: reciprocal lattice

    :type k: numpy.ndarray
    :type lattice: numpy.ndarray

    :rtype: numpy.ndarray
    """

    N = len(lattice)
    mat = np.ones((N, N))*V
    diag = np.diag(eps_0(k, lattice))
    np.fill_diagonal(mat, 0)
    mat += diag
    return mat

def calc_bandstructure(k_points, lattice, N):
    """
    Calculate the band structure of a lattice along a given k path with N samples

    :param k_points: k points
    :param lattice: reciprocal lattice
    :param N: number of samples

    :type k_points: numpy.ndarray
    :type lattice: numpy.ndarray
    :type N: int

    :rtype: numpy.ndarray
    """

    path = generate_k_path(k_points, N)
    eig_vals = np.array([np.linalg.eigvals(matrix(k, lattice)) for k in path])
    return eig_vals

def calc_moire_potential_on_grid(grid, reciprocal_moire_lattice, potential_coeffs):
    r"""
    Calculate the moire potential on a regular grid using

    .. math::
        V^{\text{M}}(\vec{r}) \approx \sum_{j=1}^6 V_j \exp\left(\text{i}\vec{G}_j^{\text{M}}\vec{r}\right)

    :param grid: :math:`\vec{r}`, a numpy meshgrid to calculate the potential on
    :param moire_lattice: :math:`\vec{G}_j^{\text{M}}`, the six moire lattice vectors
    :param potential_coeffs: :math:`V_j`, the coefficients for the potential

    :type grid: numpy.ndarray
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
    :type reciprocal_space_grid: numpy.ndarray
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

