import timeit

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from numba import cuda
from IPython import get_ipython

ipython = get_ipython()

def generate_complex_grid(r_min, r_max, i_min, i_max, r_size, i_size):
    """
    Generates a grid on the complex plane

    :param r_min: real part min
    :param r_max: real part max
    :param i_min: imaginary part min
    :param i_max: imaginary part min
    :param r_size: real part size
    :param i_size: imaginary part size

    :type r_min: float
    :type r_max: float
    :type i_min: float
    :type i_max: float
    :type r_size: float
    :type i_size: float

    :rtype: list[numpy.ndarray]
    """

    r = np.linspace(r_min, r_max, r_size)
    i = np.linspace(i_min, i_max, i_size)
    return np.meshgrid(r, 1j*i)

def diverges_after(c, n):
    """
    Returns the number of iterations it takes for the
    Mandelbrot set to diverge at point *c* or *n*.
    
    :param c: point on the complex plane
    :param n: maximum number of iterations to check for convergence

    :type c: complex128
    :type n: int64

    :rtype: int64
    """

    z = 0j
    for i in range(n):
        z = z**2 + c
        if abs(z) > 2:
            return i
    return n

mandelbrot = nb.vectorize("int64(complex128, int64)",
        target="parallel")(diverges_after)

diverges_after_gpu = cuda.jit(device=True)(diverges_after)

@cuda.jit
def mandelbrot_gpu(M, c, n):
    """
    Calculate the Mandelbrot set on the GPU

    :param M: array to save results
    :param c: grid on complex plane
    :param n: maximum number of iterations (see :func:`diverges_after`)

    :type M: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :type c: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :type n: int

    :rtype: None
    """

    i, j = cuda.grid(2) #pylint: disable=E1121,E0633

    if i < M.shape[0] and j < M.shape[1]:
        M[j, i] = diverges_after_gpu(c[j, i], n)

###################### Variable declaration ############################
grid = generate_complex_grid(-2, 1, -1.6, 1.6, 4096, 4096)
c = np.sum(grid, axis=0)

M_gpu = cuda.device_array(c.shape)
c_gpu = cuda.to_device(c)

n = 20

block = (32, 32)
grid = (M_gpu.shape[0] // block[0] if M_gpu.shape[0] % block[0] == 0
            else M_gpu.shape[0] // block[0] + 1,
        int(M_gpu.shape[0] // block[1] if M_gpu.shape[1] % block[1] == 0
            else M_gpu.shape[1] // block[1] + 1))
########################################################################

############################# Timings ##################################
# Call both functions once to trigger the jit compiler:
res_cpu = mandelbrot(c, n)
res_gpu = mandelbrot_gpu[grid, block](M_gpu, c_gpu, n) #pylint: disable=E1136

print("Pure Python timings:")
ipython.magic("timeit -n 1 -r 1 [[diverges_after(point, n) for point in row] for row in c]")
print("CPU timings:")
ipython.magic("timeit -n 1 -r 1 mandelbrot(c, n)")
print("GPU timings:")
ipython.magic("timeit -n 1 -r 1 mandelbrot_gpu[grid, block](M_gpu, c_gpu, n)")

#print(timeit.repeat(lambda: mandelbrot(c, n), repeat=7, number=10))
#print(timeit.repeat(lambda: mandelbrot_gpu[grid, block](M_gpu, c_gpu, n), repeat=7, number=10))
