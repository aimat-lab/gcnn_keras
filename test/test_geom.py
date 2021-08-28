import unittest
import sympy as sym
import numpy as np

from scipy.optimize import brentq
from scipy import special as sp

import tensorflow as tf
from kgcnn.utils.adj import get_angle_indices
from kgcnn.layers.geom import SphericalBasisLayer, NodeDistance, EdgeAngle
from kgcnn.layers.geom import BesselBasisLayer


class TestSphericalBasisLayer(unittest.TestCase):

    def test_result_original(self):

        ei = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 3], [2, 4],
                               [3, 0], [3, 1], [3, 2], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3]])
        ei1 = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                        [0, 8], [0, 9], [0, 10], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5],
                        [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [2, 0], [2, 1], [2, 3],
                        [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [3, 0],
                        [3, 1], [3, 2], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
                        [3, 10], [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [4, 6], [4, 7], [4, 8],
                        [4, 9], [4, 10], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 6], [5, 7],
                        [5, 8], [5, 10], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 7],
                        [6, 8], [6, 9], [6, 10], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5],
                        [7, 6], [7, 8], [7, 9], [7, 10], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4],
                        [8, 5], [8, 6], [8, 7], [8, 9], [8, 10], [9, 0], [9, 1], [9, 2],
                        [9, 3], [9, 4], [9, 6], [9, 7], [9, 8], [9, 10], [10, 0], [10, 1], [10, 2],
                        [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]])
        x1 = np.array([[-0.03113825,  1.54081582,  0.03192126],
               [ 0.01215347,  0.01092235, -0.01603259],
               [ 0.72169129, -0.52583353, -1.2623057 ],
               [ 0.97955987,  1.96459116,  0.03098367],
               [-0.55840223,  1.94831192, -0.83816075],
               [-0.54252252,  1.90153531,  0.93005671],
               [ 0.51522791, -0.36840234,  0.88231134],
               [-1.01070641, -0.38456999,  0.02051783],
               [ 1.7585121 , -0.17376585, -1.30871516],
               [ 0.74087192, -1.62024959, -1.27516511],
               [ 0.22023351, -0.19051179, -2.1772902 ]])
        x= np.array([[-1.26981359e-02,  1.08580416e+00,  8.00099580e-03],
               [ 2.15041600e-03, -6.03131760e-03,  1.97612040e-03],
               [ 1.01173084e+00,  1.46375116e+00,  2.76574800e-04],
               [-5.40815069e-01,  1.44752661e+00, -8.76643715e-01],
               [-5.23813634e-01,  1.43793264e+00,  9.06397294e-01]])

        ei, _, a = get_angle_indices(ei)
        ei1, _, a1 = get_angle_indices(ei1)

        rag_x = tf.RaggedTensor.from_row_lengths(np.concatenate([x,x1]), np.array([len(x),len(x1)]))
        rag_ei = tf.RaggedTensor.from_row_lengths(np.concatenate([ei,ei1]), np.array([len(ei),len(ei1)]))
        rag_a = tf.RaggedTensor.from_row_lengths(np.concatenate([a,a1]), np.array([len(a),len(a1)]))


        class Envelope(tf.keras.layers.Layer):
            """
            Envelope function that ensures a smooth cutoff
            """
            def __init__(self, exponent, name='envelope', **kwargs):
                super().__init__(name=name, **kwargs)
                self.exponent = exponent

                self.p = exponent + 1
                self.a = -(self.p + 1) * (self.p + 2) / 2
                self.b = self.p * (self.p + 2)
                self.c = -self.p * (self.p + 1) / 2

            def call(self, inputs):

                # Envelope function divided by r
                env_val = 1 / inputs + self.a * inputs**(self.p - 1) + self.b * inputs**self.p + self.c * inputs**(self.p + 1)

                return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))


        def Jn(r, n):
            """
            numerical spherical bessel functions of order n
            """
            return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)


        def Jn_zeros(n, k):
            """
            Compute the first k zeros of the spherical bessel functions up to order n (excluded)
            """
            zerosj = np.zeros((n, k), dtype="float32")
            zerosj[0] = np.arange(1, k + 1) * np.pi
            points = np.arange(1, k + n) * np.pi
            racines = np.zeros(k + n - 1, dtype="float32")
            for i in range(1, n):
                for j in range(k + n - 1 - i):
                    foo = brentq(Jn, points[j], points[j + 1], (i,))
                    racines[j] = foo
                points = racines
                zerosj[i][:k] = racines[:k]

            return zerosj


        def spherical_bessel_formulas(n):
            """
            Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
            """
            x = sym.symbols('x')

            f = [sym.sin(x)/x]
            a = sym.sin(x)/x
            for i in range(1, n):
                b = sym.diff(a, x)/x
                f += [sym.simplify(b*(-x)**i)]
                a = sym.simplify(b)
            return f

        def bessel_basis(n, k):
            """
            Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
            order n (excluded) and maximum frequency k (excluded).
            """

            zeros = Jn_zeros(n, k)
            normalizer = []
            for order in range(n):
                normalizer_tmp = []
                for i in range(k):
                    normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
                normalizer_tmp = 1/np.array(normalizer_tmp)**0.5
                normalizer += [normalizer_tmp]

            f = spherical_bessel_formulas(n)
            x = sym.symbols('x')
            bess_basis = []
            for order in range(n):
                bess_basis_tmp = []
                for i in range(k):
                    bess_basis_tmp += [sym.simplify(normalizer[order][i]*f[order].subs(x, zeros[order, i]*x))]
                bess_basis += [bess_basis_tmp]
            return bess_basis


        def sph_harm_prefactor(l, m):
            """
            Computes the constant pre-factor for the spherical harmonic of degree l and order m
            input:
            l: int, l>=0
            m: int, -l<=m<=l
            """
            return ((2*l+1) * np.math.factorial(l-abs(m)) / (4*np.pi*np.math.factorial(l+abs(m))))**0.5


        def associated_legendre_polynomials(l, zero_m_only=True):
            """
            Computes sympy formulas of the associated legendre polynomials up to order l (excluded).
            """
            z = sym.symbols('z')
            P_l_m = [[0]*(j+1) for j in range(l)]

            P_l_m[0][0] = 1
            if l > 0:
                P_l_m[1][0] = z

                for j in range(2, l):
                    P_l_m[j][0] = sym.simplify(
                        ((2*j-1)*z*P_l_m[j-1][0] - (j-1)*P_l_m[j-2][0])/j)
                if not zero_m_only:
                    for i in range(1, l):
                        P_l_m[i][i] = sym.simplify((1-2*i)*P_l_m[i-1][i-1])
                        if i + 1 < l:
                            P_l_m[i+1][i] = sym.simplify((2*i+1)*z*P_l_m[i][i])
                        for j in range(i + 2, l):
                            P_l_m[j][i] = sym.simplify(
                                ((2*j-1) * z * P_l_m[j-1][i] - (i+j-1) * P_l_m[j-2][i]) / (j - i))

            return P_l_m


        def real_sph_harm(l, zero_m_only=True, spherical_coordinates=True):
            """
            Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
            Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
            """
            if not zero_m_only:
                S_m = [0]
                C_m = [1]
                for i in range(1, l):
                    x = sym.symbols('x')
                    y = sym.symbols('y')
                    S_m += [x*S_m[i-1] + y*C_m[i-1]]
                    C_m += [x*C_m[i-1] - y*S_m[i-1]]

            P_l_m = associated_legendre_polynomials(l, zero_m_only)
            if spherical_coordinates:
                theta = sym.symbols('theta')
                z = sym.symbols('z')
                for i in range(len(P_l_m)):
                    for j in range(len(P_l_m[i])):
                        if type(P_l_m[i][j]) != int:
                            P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
                if not zero_m_only:
                    phi = sym.symbols('phi')
                    for i in range(len(S_m)):
                        S_m[i] = S_m[i].subs(x, sym.sin(
                            theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
                    for i in range(len(C_m)):
                        C_m[i] = C_m[i].subs(x, sym.sin(
                            theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))

            Y_func_l_m = [['0']*(2*j + 1) for j in range(l)]
            for i in range(l):
                Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

            if not zero_m_only:
                for i in range(1, l):
                    for j in range(1, i + 1):
                        Y_func_l_m[i][j] = sym.simplify(
                            2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
                for i in range(1, l):
                    for j in range(1, i + 1):
                        Y_func_l_m[i][-j] = sym.simplify(
                            2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

            return Y_func_l_m


        class SphericalBasisLayerOrginal(tf.keras.layers.Layer):
            def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5,
                         name='spherical_basis', **kwargs):
                super().__init__(name=name, **kwargs)

                assert num_radial <= 64
                self.num_radial = num_radial
                self.num_spherical = num_spherical

                self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
                self.envelope = Envelope(envelope_exponent)

                # retrieve formulas
                self.bessel_formulas = bessel_basis(num_spherical, num_radial)
                self.sph_harm_formulas = real_sph_harm(num_spherical)
                self.sph_funcs = []
                self.bessel_funcs = []

                # convert to tensorflow functions
                x = sym.symbols('x')
                theta = sym.symbols('theta')
                for i in range(num_spherical):
                    if i == 0:
                        first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0], 'tensorflow')(0)
                        self.sph_funcs.append(lambda tensor: tf.zeros_like(tensor) + first_sph)
                    else:
                        self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0], 'tensorflow'))
                    for j in range(num_radial):
                        self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j], 'tensorflow'))

            def call(self, inputs):
                d, Angles, id_expand_kj = inputs

                d_scaled = d * self.inv_cutoff
                rbf = [f(d_scaled) for f in self.bessel_funcs]
                rbf = tf.stack(rbf, axis=1)

                d_cutoff = self.envelope(d_scaled)
                rbf_env = d_cutoff[:, None] * rbf
                rbf_env = tf.gather(rbf_env, id_expand_kj)

                cbf = [f(Angles) for f in self.sph_funcs]
                cbf = tf.stack(cbf, axis=1)
                cbf = tf.repeat(cbf, self.num_radial, axis=1)

                return cbf*rbf_env


        dist = NodeDistance()([rag_x, rag_ei])
        angs = EdgeAngle()([rag_x, rag_ei, rag_a])
        bessel = SphericalBasisLayer(10,10,5.0)([dist, angs, rag_a])

        test = np.array(SphericalBasisLayerOrginal(10,10,5)([dist[0,:,0], angs[0,:,0], rag_a[0,:,1]]))
        test1 = np.array(SphericalBasisLayerOrginal(10,10,5)([dist[1,:,0], angs[1,:,0], rag_a[1,:,1]]))
        pred = np.array(bessel[0])
        pred1 = np.array(bessel[1])

        valid = np.abs(test - pred)
        valid1 = np.abs(test1 - pred1)
        # print(np.max(valid))
        # print(np.max(valid1))

        # For small argument and large order there are some rof errors
        # So we only have < 0.05 here
        self.assertTrue(np.max(valid) < 0.05)
        self.assertTrue(np.max(valid1) < 0.05)


class TestBesselBasisLayer(unittest.TestCase):

    def test_result_original(self):

        ei = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 3], [2, 4],
                       [3, 0], [3, 1], [3, 2], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3]])
        ei1 = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                        [0, 8], [0, 9], [0, 10], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5],
                        [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [2, 0], [2, 1], [2, 3],
                        [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [3, 0],
                        [3, 1], [3, 2], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
                        [3, 10], [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [4, 6], [4, 7], [4, 8],
                        [4, 9], [4, 10], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 6], [5, 7],
                        [5, 8], [5, 10], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 7],
                        [6, 8], [6, 9], [6, 10], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5],
                        [7, 6], [7, 8], [7, 9], [7, 10], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4],
                        [8, 5], [8, 6], [8, 7], [8, 9], [8, 10], [9, 0], [9, 1], [9, 2],
                        [9, 3], [9, 4], [9, 6], [9, 7], [9, 8], [9, 10], [10, 0], [10, 1], [10, 2],
                        [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]])
        x1 = np.array([[-0.03113825,  1.54081582,  0.03192126],
               [ 0.01215347,  0.01092235, -0.01603259],
               [ 0.72169129, -0.52583353, -1.2623057 ],
               [ 0.97955987,  1.96459116,  0.03098367],
               [-0.55840223,  1.94831192, -0.83816075],
               [-0.54252252,  1.90153531,  0.93005671],
               [ 0.51522791, -0.36840234,  0.88231134],
               [-1.01070641, -0.38456999,  0.02051783],
               [ 1.7585121 , -0.17376585, -1.30871516],
               [ 0.74087192, -1.62024959, -1.27516511],
               [ 0.22023351, -0.19051179, -2.1772902 ]])
        x= np.array([[-1.26981359e-02,  1.08580416e+00,  8.00099580e-03],
               [ 2.15041600e-03, -6.03131760e-03,  1.97612040e-03],
               [ 1.01173084e+00,  1.46375116e+00,  2.76574800e-04],
               [-5.40815069e-01,  1.44752661e+00, -8.76643715e-01],
               [-5.23813634e-01,  1.43793264e+00,  9.06397294e-01]])


        rag_x = tf.RaggedTensor.from_row_lengths(np.concatenate([x,x1]), np.array([len(x),len(x1)]))
        rag_ei = tf.RaggedTensor.from_row_lengths(np.concatenate([ei,ei1]), np.array([len(ei),len(ei1)]))
        dist = NodeDistance()([rag_x, rag_ei])
        bessel = BesselBasisLayer(10,5.0)(dist)

        class Envelope(tf.keras.layers.Layer):
            """
            Envelope function that ensures a smooth cutoff
            """
            def __init__(self, exponent, name='envelope', **kwargs):
                super().__init__(name=name, **kwargs)
                self.exponent = exponent

                self.p = exponent + 1
                self.a = -(self.p + 1) * (self.p + 2) / 2
                self.b = self.p * (self.p + 2)
                self.c = -self.p * (self.p + 1) / 2

            def call(self, inputs):

                # Envelope function divided by r
                env_val = 1 / inputs + self.a * inputs**(self.p - 1) + self.b * inputs**self.p + self.c * inputs**(self.p + 1)

                return tf.where(inputs < 1, env_val, tf.zeros_like(inputs))


        class BesselBasisLayerOriginal(tf.keras.layers.Layer):
            def __init__(self, num_radial, cutoff, envelope_exponent=5,
                         name='bessel_basis', **kwargs):
                super().__init__(name=name, **kwargs)
                self.num_radial = num_radial
                self.inv_cutoff = tf.constant(1 / cutoff, dtype=tf.float32)
                self.envelope = Envelope(envelope_exponent)

                # Initialize frequencies at canonical positions
                def freq_init(shape, dtype):
                    return tf.constant(np.pi * np.arange(1, shape + 1, dtype=np.float32), dtype=dtype)
                self.frequencies = self.add_weight(name="frequencies", shape=self.num_radial,
                                                   dtype=tf.float32, initializer=freq_init, trainable=True)

            def call(self, inputs):
                d_scaled = inputs * self.inv_cutoff

                # Necessary for proper broadcasting behaviour
                d_scaled = tf.expand_dims(d_scaled, -1)

                d_cutoff = self.envelope(d_scaled)
                return d_cutoff * tf.sin(self.frequencies * d_scaled)

        test0 = BesselBasisLayerOriginal(10,5.0)(dist[0,:,0])
        test1 = BesselBasisLayerOriginal(10,5.0)(dist[1,:,0])

        self.assertTrue(np.max(np.abs(test0 - bessel[0])) < 1e-5)
        self.assertTrue(np.max(np.abs(test1 - bessel[1])) < 1e-5)


if __name__ == '__main__':
    unittest.main()
