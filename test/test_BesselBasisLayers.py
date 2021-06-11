import unittest

import numpy as np

import tensorflow as tf
from kgcnn.data.mol.methods import get_angle_indices
from kgcnn.layers.geom import BesselBasisLayer, NodeDistance


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


# x = tf.RaggedTensor.from_row_lengths(np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0],[1,0,0]], dtype=np.float32), np.array([3,3],dtype=np.int64))
# edi = tf.RaggedTensor.from_row_lengths(np.array([[0,1],[0,2],[1,0],[2,0],[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]],dtype=np.int64), np.array([4,6],dtype=np.int64))
# adi = tf.RaggedTensor.from_row_lengths(np.array([[0,1],[0,3],[1,0],[1,2],[2,1],[2,3],[3,0],[3,2],
#                                                  [0,1],[0,3],[0,4],[0,5],
#                                                  [1,0],[1,2],[1,3],[1,5],
#                                                  [2,1],[2,3],[2,4],[2,5],
#                                                  [3,0],[3,1],[3,2],[3,4],
#                                                  [4,0],[4,2],[4,3],[4,5],
#                                                  [5,0],[5,1],[5,2],[5,4],
#                                                  ],dtype=np.int64), np.array([8,24],dtype=np.int64))
