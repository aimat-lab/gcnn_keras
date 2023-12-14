import numpy as np
import scipy as sp
import scipy.special
from keras import ops
from keras.layers import Layer
from scipy.optimize import brentq


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
        x (Tensor): Values to compute :math:`j_n(x)` for.
        n (int): Positive integer for the bessel order :math:`n`.

    Returns:
        Tensor: Spherical bessel function of order :math:`n`
    """
    sin_x = ops.sin(x - n * np.pi / 2)
    cos_x = ops.cos(x - n * np.pi / 2)
    sum_sin = ops.zeros_like(x)
    sum_cos = ops.zeros_like(x)
    for k in range(int(np.floor(n / 2)) + 1):
        if 2 * k < n + 1:
            prefactor_sin = float(sp.special.factorial(n + 2 * k) / np.power(2, 2 * k) / sp.special.factorial(
                2 * k) / sp.special.factorial(n - 2 * k) * np.power(-1, k))
            sum_sin += prefactor_sin * ops.power(x, - (2 * k + 1))
    for k in range(int(np.floor((n - 1) / 2)) + 1):
        if 2 * k + 1 < n + 1:
            prefactor_cos = float(sp.special.factorial(n + 2 * k + 1) / np.power(2, 2 * k + 1) / sp.special.factorial(
                2 * k + 1) / sp.special.factorial(n - 2 * k - 1) * np.power(-1, k))
            sum_cos += prefactor_cos * ops.power(x, - (2 * k + 2))
    return sum_sin * sin_x + sum_cos * cos_x


class SphericalBesselJnExplicit(Layer):
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
    """

    def __init__(self, n=0, fused: bool = False, **kwargs):
        r"""Initialize layer with constant n.

        Args:
            n (int): Positive integer for the bessel order :math:`n`.
            fused (bool): Whether to compute polynomial in a fused tensor representation.
        """
        super(SphericalBesselJnExplicit, self).__init__(**kwargs)
        self.n = n
        self.fused = fused
        self._pre_factor_sin = []
        self._pre_factor_cos = []
        self._powers_sin = []
        self._powers_cos = []

        for k in range(int(np.floor(n / 2)) + 1):
            if 2 * k < n + 1:
                fac_sin = float(sp.special.factorial(n + 2 * k) / np.power(2, 2 * k) / sp.special.factorial(
                    2 * k) / sp.special.factorial(n - 2 * k) * np.power(-1, k))
                pow_sin = - (2 * k + 1)
                self._pre_factor_sin.append(fac_sin)
                self._powers_sin.append(pow_sin)

        for k in range(int(np.floor((n - 1) / 2)) + 1):
            if 2 * k + 1 < n + 1:
                fac_cos = float(sp.special.factorial(n + 2 * k + 1) / np.power(2, 2 * k + 1) / sp.special.factorial(
                        2 * k + 1) / sp.special.factorial(n - 2 * k - 1) * np.power(-1, k))
                pow_cos = - (2 * k + 2)
                self._pre_factor_cos.append(fac_cos)
                self._powers_cos.append(pow_cos)

        if self.fused:
            self._pre_factor_sin = ops.convert_to_tensor(self._pre_factor_sin, dtype=self.dtype)
            self._pre_factor_cos = ops.convert_to_tensor(self._pre_factor_cos, dtype=self.dtype)
            self._powers_sin = ops.convert_to_tensor(self._powers_sin, dtype=self.dtype)
            self._powers_cos = ops.convert_to_tensor(self._powers_cos, dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(SphericalBesselJnExplicit, self).build(input_shape)

    def call(self, x, **kwargs):
        """Element-wise operation.

        Args:
            x (Tensor): Values to compute :math:`j_n(x)` for.

        Returns:
            Tensor: Spherical bessel function of order :math:`n`
        """
        n = self.n
        sin_x = ops.sin(x - n * np.pi / 2)
        cos_x = ops.cos(x - n * np.pi / 2)
        if not self.fused:
            sum_sin = ops.zeros_like(x)
            sum_cos = ops.zeros_like(x)
            for a, r in zip(self._pre_factor_sin, self._powers_sin):
                sum_sin += a * ops.power(x, r)
            for b, s in zip(self._pre_factor_cos, self._powers_cos):
                sum_cos += b * ops.power(x, s)
        else:
            sum_sin = ops.sum(self._pre_factor_sin * ops.power(ops.expand_dims(x, axis=-1), self._powers_sin), axis=-1)
            sum_cos = ops.sum(self._pre_factor_cos * ops.power(ops.expand_dims(x, axis=-1), self._powers_cos), axis=-1)
        return sum_sin * sin_x + sum_cos * cos_x

    def get_config(self):
        """Update layer config."""
        config = super(SphericalBesselJnExplicit, self).get_config()
        config.update({"n": self.n})
        return config


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
        x (Tensor): Values to compute :math:`j_n(x)` for.
        n (int): Positive integer for the bessel order :math:`n`.

    Returns:
        Tensor: Spherical bessel function of order :math:`n`
    """
    if n < 0:
        raise ValueError("Order parameter must be >= 0 for this implementation of spherical bessel function.")
    if n == 0:
        return ops.sin(x) / x
    elif n == 1:
        return ops.sin(x) / ops.square(x) - ops.cos(x) / x
    else:
        j_n = ops.sin(x) / x
        j_nn = ops.sin(x) / ops.square(x) - ops.cos(x) / x
        for i in range(1, n):
            temp = j_nn
            j_nn = (2 * i + 1) / x * j_nn - j_n
            j_n = temp
        return j_nn


def tf_legendre_polynomial_pn(x, n=0):
    r"""Compute the (non-associated) Legendre polynomial :math:`P_n(x)` for constant positive integer :math:`n`
    via explicit formula.
    TensorFlow has to cache the function for each :math:`n`. No gradient through :math:`n` or very large number
    of :math:`n` is possible.
    Closed form can be viewed at https://en.wikipedia.org/wiki/Legendre_polynomials.

    :math:`P_n(x)=\sum_{k=0}^{\lfloor n/2\rfloor} (-1)^k \frac{(2n - 2k)! \, }{(n-k)! \, (n-2k)! \, k! \, 2^n} x^{n-2k}`

    Args:
        x (Tensor): Values to compute :math:`P_n(x)` for.
        n (int): Positive integer for :math:`n` in :math:`P_n(x)`.

    Returns:
        Tensor: Legendre Polynomial of order :math:`n`.
    """
    out_sum = ops.zeros_like(x)
    prefactors = [
        float((-1) ** k * sp.special.factorial(2 * n - 2 * k) / sp.special.factorial(n - k) / sp.special.factorial(
            n - 2 * k) / sp.special.factorial(k) / 2 ** n) for k in range(0, int(np.floor(n / 2)) + 1)]
    powers = [float(n - 2 * k) for k in range(0, int(np.floor(n / 2)) + 1)]
    for i in range(len(powers)):
        out_sum = out_sum + prefactors[i] * ops.power(x, powers[i])
    return out_sum


class LegendrePolynomialPn(Layer):
    r"""Compute the (non-associated) Legendre polynomial :math:`P_n(x)` for constant positive integer :math:`n`
    via explicit formula.
    TensorFlow has to cache the function for each :math:`n`. No gradient through :math:`n` or very large number
    of :math:`n` is possible.
    Closed form can be viewed at https://en.wikipedia.org/wiki/Legendre_polynomials.

    :math:`P_n(x)=\sum_{k=0}^{\lfloor n/2\rfloor} (-1)^k \frac{(2n - 2k)! \, }{(n-k)! \, (n-2k)! \, k! \, 2^n} x^{n-2k}`

    """

    def __init__(self, n=0, fused: bool = False, **kwargs):
        r"""Initialize layer with constant n.

        Args:
            n (int): Positive integer for :math:`n` in :math:`P_n(x)`.
            fused (bool): Whether to compute polynomial in a fused tensor representation.
        """
        super(LegendrePolynomialPn, self).__init__(**kwargs)
        self.fused = fused
        self.n = n
        self._pre_factors = [
            float((-1) ** k * sp.special.factorial(2 * n - 2 * k) / sp.special.factorial(n - k) / sp.special.factorial(
                n - 2 * k) / sp.special.factorial(k) / 2 ** n) for k in range(0, int(np.floor(n / 2)) + 1)
        ]
        self._powers = [float(n - 2 * k) for k in range(0, int(np.floor(n / 2)) + 1)]
        if self.fused:
            # Or maybe also as weight.
            self._powers = ops.convert_to_tensor(self._powers, dtype=self.dtype)
            self._pre_factors = ops.convert_to_tensor(self._pre_factors, dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(LegendrePolynomialPn, self).build(input_shape)

    def call(self, x, **kwargs):
        """Element-wise operation.

        Args:
            x (Tensor): Values to compute :math:`P_n(x)` for.

        Returns:
            Tensor: Legendre Polynomial of order :math:`n`.
        """
        if not self.fused:
            out_sum = ops.zeros_like(x)
            for a, r in zip(self._pre_factors, self._powers):
                out_sum = out_sum + a * ops.power(x, r)
        else:
            out_sum = ops.sum(self._pre_factors * ops.power(ops.expand_dims(x, axis=-1), self._powers), axis=-1)
        return out_sum

    def get_config(self):
        """Update layer config."""
        config = super(LegendrePolynomialPn, self).get_config()
        config.update({"n": self.n, "fused": self.fused})
        return config


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
        theta (Tensor): Values to compute :math:`Y_l(\cos\theta)` for.
        l (int): Positive integer for :math:`l` in :math:`Y_l(\cos\theta)`.

    Returns:
        Tensor: Spherical harmonics for :math:`m=0` and constant non-integer :math:`l`.
    """
    x = ops.cos(theta)
    out_sum = ops.zeros_like(x)
    prefactors = [
        float((-1) ** k * sp.special.factorial(2 * l - 2 * k) / sp.special.factorial(l - k) / sp.special.factorial(
            l - 2 * k) / sp.special.factorial(k) / 2 ** l) for k in range(0, int(np.floor(l / 2)) + 1)]
    powers = [float(l - 2 * k) for k in range(0, int(np.floor(l / 2)) + 1)]
    for i in range(len(powers)):
        out_sum = out_sum + prefactors[i] * ops.power(x, powers[i])
    out_sum = out_sum * float(np.sqrt((2 * l + 1) / 4 / np.pi))
    return out_sum


class SphericalHarmonicsYl(Layer):
    r"""Compute the spherical harmonics :math:`Y_{ml}(\cos\theta)` for :math:`m=0` and constant non-integer :math:`l`.
    TensorFlow has to cache the function for each :math:`l`. No gradient through :math:`l` or very large number
    of :math:`n` is possible. Uses a simplified formula with :math:`m=0` from
    https://en.wikipedia.org/wiki/Spherical_harmonics:

    :math:`Y_{l}^{m}(\theta ,\phi)=\sqrt{\frac{(2l+1)}{4\pi} \frac{(l -m)!}{(l +m)!}} \, P_{l}^{m}(\cos{\theta }) \,
    e^{i m \phi}`

    where the associated Legendre polynomial simplifies to :math:`P_l(x)` for :math:`m=0`:

    :math:`P_n(x)=\sum_{k=0}^{\lfloor n/2\rfloor} (-1)^k \frac{(2n - 2k)! \, }{(n-k)! \, (n-2k)! \, k! \, 2^n} x^{n-2k}`

    """

    def __init__(self, l=0, fused: bool = False, **kwargs):
        r"""Initialize layer with constant l.

        Args:
            l (int): Positive integer for :math:`l` in :math:`Y_l(\cos\theta)`.
            fused (bool): Whether to compute polynomial in a fused tensor representation.
        """
        super(SphericalHarmonicsYl, self).__init__(**kwargs)
        self.l = l
        self.fused = fused
        self._pre_factors = [
            float((-1) ** k * sp.special.factorial(2 * l - 2 * k) / sp.special.factorial(l - k) / sp.special.factorial(
                l - 2 * k) / sp.special.factorial(k) / 2 ** l) for k in range(0, int(np.floor(l / 2)) + 1)]
        self._powers = [float(l - 2 * k) for k in range(0, int(np.floor(l / 2)) + 1)]
        self._scale = float(np.sqrt((2 * l + 1) / 4 / np.pi))
        if self.fused:
            # Or maybe also as weight.
            self._powers = ops.convert_to_tensor(self._powers, dtype=self.dtype)
            self._pre_factors = ops.convert_to_tensor(self._pre_factors, dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(SphericalHarmonicsYl, self).build(input_shape)

    def call(self, theta, **kwargs):
        """Element-wise operation.

        Args:
            theta (Tensor): Values to compute :math:`Y_l(\cos\theta)` for.

        Returns:
            Tensor: Spherical harmonics for :math:`m=0` and constant non-integer :math:`l`.
        """
        x = ops.cos(theta)
        if not self.fused:
            out_sum = ops.zeros_like(x)
            for a, r in zip(self._pre_factors, self._powers):
                out_sum = out_sum + a * ops.power(x, r)
        else:
            out_sum = ops.sum(self._pre_factors * ops.power(ops.expand_dims(x, axis=-1), self._powers), axis=-1)
        out_sum = out_sum * self._scale
        return out_sum

    def get_config(self):
        """Update layer config."""
        config = super(SphericalHarmonicsYl, self).get_config()
        config.update({"l": self.l, "fused": self.fused})
        return config


def tf_associated_legendre_polynomial(x, l=0, m=0):
    r"""Compute the associated Legendre polynomial :math:`P_{l}^{m}(x)` for :math:`m` and constant positive
    integer :math:`l` via explicit formula.
    Closed Form from taken from https://en.wikipedia.org/wiki/Associated_Legendre_polynomials.

    :math:`P_{l}^{m}(x)=(-1)^{m}\cdot 2^{l}\cdot (1-x^{2})^{m/2}\cdot \sum_{k=m}^{l}\frac{k!}{(k-m)!}\cdot x^{k-m}
    \cdot \binom{l}{k}\binom{\frac{l+k-1}{2}}{l}`.

    Args:
        x (Tensor): Values to compute :math:`P_{l}^{m}(x)` for.
        l (int): Positive integer for :math:`l` in :math:`P_{l}^{m}(x)`.
        m (int): Positive/Negative integer for :math:`m` in :math:`P_{l}^{m}(x)`.

    Returns:
        Tensor: Legendre Polynomial of order n.
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

    x_prefactor = ops.power(1 - ops.square(x), m / 2) * float(np.power(-1, m) * np.power(2, l))
    sum_out = ops.zeros_like(x)
    for k in range(m, l + 1):
        sum_out += ops.power(x, k - m) * float(
            sp.special.factorial(k) / sp.special.factorial(k - m) * sp.special.binom(l, k) *
            sp.special.binom((l + k - 1) / 2, l))

    return sum_out * x_prefactor * neg_m


class AssociatedLegendrePolynomialPlm(Layer):
    r"""Compute the associated Legendre polynomial :math:`P_{l}^{m}(x)` for :math:`m` and constant positive
    integer :math:`l` via explicit formula.
    Closed Form from taken from https://en.wikipedia.org/wiki/Associated_Legendre_polynomials.

    :math:`P_{l}^{m}(x)=(-1)^{m}\cdot 2^{l}\cdot (1-x^{2})^{m/2}\cdot \sum_{k=m}^{l}\frac{k!}{(k-m)!}\cdot x^{k-m}
    \cdot \binom{l}{k}\binom{\frac{l+k-1}{2}}{l}`.
    """

    def __init__(self, l: int = 0, m: int = 0, fused: bool = False, **kwargs):
        r"""Initialize layer with constant m, l.

        Args:
            l (int): Positive integer for :math:`l` in :math:`P_{l}^{m}(x)`.
            m (int): Positive/Negative integer for :math:`m` in :math:`P_{l}^{m}(x)`.
            fused (bool): Whether to compute polynomial in a fused tensor representation.
        """
        super(AssociatedLegendrePolynomialPlm, self).__init__(**kwargs)
        self.m = m
        self.fused = fused
        self.l = l
        if np.abs(m) > l:
            raise ValueError("Error: Legendre polynomial must have -l<= m <= l")
        if l < 0:
            raise ValueError("Error: Legendre polynomial must have l>=0")
        if m < 0:
            m = -m
            neg_m = float(np.power(-1, m) * sp.special.factorial(l - m) / sp.special.factorial(l + m))
        else:
            neg_m = 1
        self._m = m
        self._neg_m = neg_m
        self._x_pre_factor = float(np.power(-1, m) * np.power(2, l))
        self._powers = []
        self._pre_factors = []
        for k in range(m, l + 1):
            self._powers.append(k - m)
            fac = float(
                sp.special.factorial(k) / sp.special.factorial(k - m) * sp.special.binom(l, k) *
                sp.special.binom((l + k - 1) / 2, l))
            self._pre_factors.append(fac)

        if self.fused:
            # Or maybe also as weight.
            self._powers = ops.convert_to_tensor(self._powers, dtype=self.dtype)
            self._pre_factors = ops.convert_to_tensor(self._pre_factors, dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(AssociatedLegendrePolynomialPlm, self).build(input_shape)

    def call(self, x, **kwargs):
        """Element-wise operation.

        Args:
            x (Tensor): Values to compute :math:`P_{l}^{m}(x)` for.

        Returns:
            Tensor: Legendre Polynomial of order n.
        """
        neg_m = self._neg_m
        m = self._m
        x_pre_factor = ops.power(1 - ops.square(x), m / 2) * self._x_pre_factor
        if not self.fused:
            sum_out = ops.zeros_like(x)
            for a, r in zip(self._pre_factors, self._powers):
                sum_out += ops.power(x, r) * a
        else:
            sum_out = ops.sum(self._pre_factors * ops.power(ops.expand_dims(x, axis=-1), self._powers), axis=-1)
        return sum_out * x_pre_factor * neg_m

    def get_config(self):
        """Update layer config."""
        config = super(AssociatedLegendrePolynomialPlm, self).get_config()
        config.update({"l": self.l, "m": self.m, "fused": self.fused})
        return config
