import numpy as np

from bandcalc.constants import unit_prefixes

def get_unit_prefix(data):
    """
    Suggest a unit prefix for given *data*, to make it more readable

    :param data: find unit prefix for this data

    :type data: numpy.ndarray

    :rtype: tuple(numpy.ndarray, str)
    """

    max = np.abs(data).max()

    if max == 0:
        return (data, "")

    exponent = int(np.floor(np.log(max)/np.log(1000)))*3

    if exponent in unit_prefixes:
        data *= 10**(-exponent)
        prefix = unit_prefixes[exponent]
        return (data, prefix)

    return (data, "")

