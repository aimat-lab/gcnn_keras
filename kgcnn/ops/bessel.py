import numpy as np
from scipy.optimize import brentq
import scipy as sp
import scipy.special


def spherical_jn(r, n):
    """Compute spherical Bessel function jn(r) via scipy.

    Args:
        r (array_like): Argument
        n (array_like): Order.

    Returns:
        jn_r: Values of the spherical Bessel function
    """
    return np.sqrt(np.pi/(2*r)) * sp.special.jv(n+0.5, r)


def spherical_jn_zeros(n, k):
    """Compute the first k zeros of the spherical bessel functions up to order n (excluded).
    Taken from https://github.com/klicperajo/dimenet.

    Args:
        n: Order.
        k: Number of zero crossings.

    Returns:
        zeros_jn: List of zero crossings of shape (n,k)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(spherical_jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def normalization_prefactor():
    pass