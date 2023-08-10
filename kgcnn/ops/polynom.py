import numpy as np
import scipy as sp
import scipy.special
import tensorflow as tf
from scipy.optimize import brentq


@tf.function
def tf_spherical_bessel_jn_explicit(x, n=0):
    r"""Compute spherical bessel functions :math:`j_n(x)` for constant positive integer :math:`n` explicitly.
    TensorFlow has to cache the function for each :math:`n`. No gradient through :math:`n` or very large number
    of :math:`n`'s is possible.
    The spherical bessel functions and there properties can be looked up at
    https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions.
    For this implementation the explicit expression from https://dlmf.nist.gov/10.49 has been used.
    The definition is:

    :math:`a_{k}(n+\tfrac{1}{2})=\begin{cases}\dfrac{(n+k)!}{2^{k}k!(n-k)!},&k=0,1,\dotsc,n\\
    0,&k=n+1,n+2,\dotsc\end{cases}`

    :math:`\mathsf{j}_{n}\left(z\right)=\sin\left(z-\tfrac{1}{2}n\pi\right)\sum_{k=0}^{\left\lfloor n/2\right\rfloor}
    (-1)^{k}\frac{a_{2k}(n+\tfrac{1}{2})}{z^{2k+1}}+\cos\left(z-\tfrac{1}{2}n\pi\right)
    \sum_{k=0}^{\left\lfloor(n-1)/2\right\rfloor}(-1)^{k}\frac{a_{2k+1}(n+\tfrac{1}{2})}{z^{2k+2}}.`

    Args:
        x (tf.Tensor): Values to compute :math:`j_n(x)` for.
        n (int): Positive integer for the bessel order :math:`n`.

    Returns:
        tf.Tensor: Spherical bessel function of order :math:`n`
    """
    sin_x = tf.sin(x - n * np.pi / 2)
    cos_x = tf.cos(x - n * np.pi / 2)
    sum_sin = tf.zeros_like(x)
    sum_cos = tf.zeros_like(x)
    for k in range(int(np.floor(n / 2)) + 1):
        if 2 * k < n + 1:
            prefactor = float(sp.special.factorial(n + 2 * k) / np.power(2, 2 * k) / sp.special.factorial(
                2 * k) / sp.special.factorial(n - 2 * k) * np.power(-1, k))
            sum_sin += prefactor * tf.pow(x, - (2 * k + 1))
    for k in range(int(np.floor((n - 1) / 2)) + 1):
        if 2 * k + 1 < n + 1:
            prefactor = float(sp.special.factorial(n + 2 * k + 1) / np.power(2, 2 * k + 1) / sp.special.factorial(
                2 * k + 1) / sp.special.factorial(n - 2 * k - 1) * np.power(-1, k))
            sum_cos += prefactor * tf.pow(x, - (2 * k + 2))
    return sum_sin * sin_x + sum_cos * cos_x


@tf.function
def tf_spherical_bessel_jn(x, n=0):
    r"""Compute spherical bessel functions :math:`j_n(x)` for constant positive integer :math:`n` via recursion.
    TensorFlow has to cache the function for each :math:`n`. No gradient through :math:`n` or very large number
    of :math:`n` is possible.
    The spherical bessel functions and there properties can be looked up at
    https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions.
    The recursive rule is constructed from https://dlmf.nist.gov/10.51. The recursive definition is:

    :math:`j_{n+1}(z)=((2n+1)/z)j_{n}(z)-j_{n-1}(z)`

    :math:`j_{0}(x)=\frac{\sin x}{x}`

    :math:`j_{1}(x)=\frac{1}{x}\frac{\sin x}{x} - \frac{\cos x}{x}`

    :math:`j_{2}(x)=\left(\frac{3}{x^{2}} - 1\right)\frac{\sin x}{x} - \frac{3}{x}\frac{\cos x}{x}`

    Args:
        x (tf.Tensor): Values to compute :math:`j_n(x)` for.
        n (int): Positive integer for the bessel order :math:`n`.

    Returns:
        tf.tensor: Spherical bessel function of order :math:`n`
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
    r"""Compute the (non-associated) Legendre polynomial :math:`P_n(x)` for constant positive integer :math:`n`
    via explicit formula.
    TensorFlow has to cache the function for each :math:`n`. No gradient through :math:`n` or very large number
    of :math:`n` is possible.
    Closed form can be viewed at https://en.wikipedia.org/wiki/Legendre_polynomials.

    :math:`P_n(x)=\sum_{k=0}^{\lfloor n/2\rfloor} (-1)^k \frac{(2n - 2k)! \, }{(n-k)! \, (n-2k)! \, k! \, 2^n} x^{n-2k}`

    Args:
        x (tf.Tensor): Values to compute :math:`P_n(x)` for.
        n (int): Positive integer for :math:`n` in :math:`P_n(x)`.

    Returns:
        tf.tensor: Legendre Polynomial of order :math:`n`.
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
    r"""Compute the spherical harmonics :math:`Y_{ml}(\cos\theta)` for :math:`m=0` and constant non-integer :math:`l`.
    TensorFlow has to cache the function for each :math:`l`. No gradient through :math:`l` or very large number
    of :math:`n` is possible. Uses a simplified formula with :math:`m=0` from
    https://en.wikipedia.org/wiki/Spherical_harmonics:

    :math:`Y_{l}^{m}(\theta ,\phi)=\sqrt{\frac{(2l+1)}{4\pi} \frac{(l -m)!}{(l +m)!}} \, P_{l}^{m}(\cos{\theta }) \,
    e^{i m \phi}`

    where the associated Legendre polynomial simplifies to :math:`P_l(x)` for :math:`m=0`:

    :math:`P_n(x)=\sum_{k=0}^{\lfloor n/2\rfloor} (-1)^k \frac{(2n - 2k)! \, }{(n-k)! \, (n-2k)! \, k! \, 2^n} x^{n-2k}`

    Args:
        theta (tf.Tensor): Values to compute :math:`Y_l(\cos\theta)` for.
        l (int): Positive integer for :math:`l` in :math:`Y_l(\cos\theta)`.

    Returns:
        tf.tensor: Spherical harmonics for :math:`m=0` and constant non-integer :math:`l`.
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


@tf.function
def tf_associated_legendre_polynomial(x, l=0, m=0):
    r"""Compute the associated Legendre polynomial :math:`P_{l}^{m}(x)` for :math:`m` and constant positive
    integer :math:`l` via explicit formula.
    Closed Form from taken from https://en.wikipedia.org/wiki/Associated_Legendre_polynomials.

    :math:`P_{l}^{m}(x)=(-1)^{m}\cdot 2^{l}\cdot (1-x^{2})^{m/2}\cdot \sum_{k=m}^{l}\frac{k!}{(k-m)!}\cdot x^{k-m}
    \cdot \binom{l}{k}\binom{\frac{l+k-1}{2}}{l}`.

    Args:
        x (tf.Tensor): Values to compute :math:`P_{l}^{m}(x)` for.
        l (int): Positive integer for :math:`l` in :math:`P_{l}^{m}(x)`.
        m (int): Positive/Negative integer for :math:`m` in :math:`P_{l}^{m}(x)`.

    Returns:
        tf.tensor: Legendre Polynomial of order n.
    """
    if np.abs(m) > l:
        raise ValueError("Error: Legendre polynomial must have -l<= m <= l")
    if l < 0:
        raise ValueError("Error: Legendre polynomial must have l>=0")
    if m < 0:
        m = -m
        neg_m = float(np.power(-1, m) * sp.special.factorial(l - m) / sp.special.factorial(l + m))
    else:
        neg_m = 1

    x_prefactor = tf.pow(1 - tf.square(x), m / 2) * float(np.power(-1, m) * np.power(2, l))
    sum_out = tf.zeros_like(x)
    for k in range(m, l + 1):
        sum_out += tf.pow(x, k - m) * float(
            sp.special.factorial(k) / sp.special.factorial(k - m) * sp.special.binom(l, k) *
            sp.special.binom((l + k - 1) / 2, l))

    return sum_out * x_prefactor * neg_m


def spherical_bessel_jn(r, n):
    r"""Compute spherical Bessel function :math:`j_n(r)` via scipy.
    The spherical bessel functions and there properties can be looked up at
    https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions .

    Args:
        r (np.ndarray): Argument
        n (np.ndarray, int): Order.

    Returns:
        np.array: Values of the spherical Bessel function
    """
    return np.sqrt(np.pi / (2 * r)) * sp.special.jv(n + 0.5, r)


def spherical_bessel_jn_zeros(n, k):
    r"""Compute the first :math:`k` zeros of the spherical bessel functions :math:`j_n(r)` up to
    order :math:`n` (excluded).
    Taken from the original implementation of DimeNet at https://github.com/klicperajo/dimenet.

    Args:
        n: Order.
        k: Number of zero crossings.

    Returns:
        np.ndarray: List of zero crossings of shape (n, k)
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
    r"""Compute the normalization or rescaling pre-factor for the spherical bessel functions :math:`j_n(r)` up to
    order :math:`n` (excluded) and maximum frequency :math:`k` (excluded).
    Taken from the original implementation of DimeNet at https://github.com/klicperajo/dimenet.

    Args:
        n: Order.
        k: frequency.

    Returns:
        np.ndarray: Normalization of shape (n, k)
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
