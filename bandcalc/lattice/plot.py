import numpy as np
import matplotlib.pyplot as plt

def plot_lattice(ax, lattice, *plotargs, **plotkargs):
    ax.plot(lattice[:,0], lattice[:,1], *plotargs, **plotkargs)
    ax.axis("equal")

def plot_bandstructure(ax, bandstructure, k_point_names, N, *plotargs, **plotkargs):
    ax.plot(np.sort(bandstructure), *plotargs, **plotkargs)
    num_points = len(k_point_names)
    ax.set_xticks([i*N/(num_points-1) for i in range(num_points)])
    ax.set_xticklabels(k_point_names)

def plot_k_path(ax, path, *plotargs, **plotkargs):
    ax.plot(path[:,0], path[:,1], *plotargs, **plotkargs)

