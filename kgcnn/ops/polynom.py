import numpy as np
import scipy as sp
import scipy.special
import tensorflow as tf
from scipy.optimize import brentq


@tf.function
def tf_spherical_bessel_jn_explicit(x, n=0):
    r"""Compute spherical bessel functions $j_n(x)$ for constant positive integer $n$ explicitly.
    TensorFlow has to cache the function for each n. No gradient through n or very large number of n's is possible.
    Source: https://dlmf.nist.gov/10.49

    Args:
        x (tf.tensor): Values to compute jn(x) for.
        n (int): Positive integer for the bessel order n.

    Returns:
        tf.tensor: Spherical bessel function of order n
    """
    sin_x = tf.sin(x - n * np.pi / 2)
    cos_x = tf.cos(x - n * np.pi / 2)
    sum_sin = tf.zeros_like(x)
    sum_cos = tf.zeros_like(x)
    for k in range(int(np.floor(n / 2)) + 1):
        if 2 * k < n + 1:
            prefactor = float(sp.special.factorial(n + 2 * k) / np.power(2, 2 * k) / sp.special.factorial(
                        2 * k) / sp.special.factorial(n - 2 * k) * np.power(-1, k))
            sum_sin += prefactor*tf.pow(x, - (2*k+1))
    for k in range(int(np.floor((n - 1) / 2)) + 1):
        if 2 * k + 1 < n + 1:
            prefactor = float(sp.special.factorial(n + 2 * k + 1) / np.power(2, 2 * k + 1) / sp.special.factorial(
                        2 * k + 1) / sp.special.factorial(n - 2 * k - 1) * np.power(-1, k))
            sum_cos += prefactor * tf.pow(x, - (2 * k + 2))
    return sum_sin*sin_x + sum_cos*cos_x


@tf.function
def tf_spherical_bessel_jn(x, n=0):
    """Compute spherical bessel functions jn(x) for constant positive integer n via recursion.

    Args:
        x (tf.tensor): Values to compute jn(x) for.
        n (int): Positive integer for the bessel order n.

    Returns:
        tf.tensor: Spherical bessel function of order n
    """
    if n < 0:
        raise ValueError("Order parameter must be >= 0 for this implementation of spherical bessel function.")
    if n == 0:
        return tf.sin(x) / x
    elif n == 1:
        return tf.sin(x) / tf.square(x) - tf.cos(x) / x
    else:
        j_n = tf.sin(x) / x
        j_nn = tf.sin(x) / tf.square(x) - tf.cos(x) / x
        for i in range(1, n):
            temp = j_nn
            j_nn = (2 * i + 1) / x * j_nn - j_n
            j_n = temp
        return j_nn


@tf.function
def tf_legendre_polynomial_pn(x, n=0):
    """Compute the (non-associated) Legendre polynomial for constant positive integer n via explicit formula.

    Args:
        x (tf.tensor): Values to compute Pn(x) for.
        n (int): Positive integer for n in Pn(x).

    Returns:
        tf.tensor: Legendre Polynomial of order n.
    """
    out_sum = tf.zeros_like(x)
    prefactors = [
        float((-1) ** k * sp.special.factorial(2 * n - 2 * k) / sp.special.factorial(n - k) / sp.special.factorial(
            n - 2 * k) / sp.special.factorial(k) / 2 ** n) for k in range(0, int(np.floor(n / 2)) + 1)]
    powers = [float(n - 2 * k) for k in range(0, int(np.floor(n / 2)) + 1)]
    for i in range(len(powers)):
        out_sum = out_sum + prefactors[i] * tf.pow(x, powers[i])
    return out_sum


@tf.function
def tf_spherical_harmonics_yl(theta, l=0):
    """Compute the spherical harmonics for m=0 and constant non-integer l.

    Args:
        theta (tf.tensor): Values to compute Yl(theta) for.
        l (int): Positive integer for l in Yl(x).

    Returns:
        tf.tensor: Spherical harmonics for m=0 and constant non-integer l.
    """
    x = tf.cos(theta)
    out_sum = tf.zeros_like(x)
    prefactors = [
        float((-1) ** k * sp.special.factorial(2 * l - 2 * k) / sp.special.factorial(l - k) / sp.special.factorial(
            l - 2 * k) / sp.special.factorial(k) / 2 ** l) for k in range(0, int(np.floor(l / 2)) + 1)]
    powers = [float(l - 2 * k) for k in range(0, int(np.floor(l / 2)) + 1)]
    for i in range(len(powers)):
        out_sum = out_sum + prefactors[i] * tf.pow(x, powers[i])
    out_sum = out_sum * float(np.sqrt((2 * l + 1) / 4 / np.pi))
    return out_sum


# @tf.function
# def tf_associated_legendre_polynomial(x, l=0, m=0):
#     """Compute the associated Legendre polynomial for m and constant positive integer l via explicit formula.
#
#     Args:
#         x (tf.tensor): Values to compute Plm(x) for.
#         l (int): Positive integer for l in Plm(x).
#         m (int): Positive/Negative integer for m in Plm(x).
#
#     Returns:
#         tf.tensor: Legendre Polynomial of order n.
#     """
#     if m==0:
#     else:
#         prefactors = [ for k in range(m,l)]
#         powers = [ for k in range(m,l)]
#         if


def spherical_bessel_jn(r, n):
    """Compute spherical Bessel function jn(r) via scipy.

    Args:
        r (array_like): Argument
        n (array_like): Order.

    Returns:
        array_like: Values of the spherical Bessel function
    """
    return np.sqrt(np.pi / (2 * r)) * sp.special.jv(n + 0.5, r)


def spherical_bessel_jn_zeros(n, k):
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
            foo = brentq(spherical_bessel_jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_jn_normalization_prefactor(n, k):
    """Compute the normalization or rescaling pre-factor for the spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    Taken from https://github.com/klicperajo/dimenet.

    Args:
        n: Order.
        k: frequency.

    Returns:
        norm: Normalization of shape (n, k)
    """
    zeros = spherical_bessel_jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * spherical_bessel_jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]
    return np.array(normalizer)
