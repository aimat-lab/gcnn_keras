import unittest
import numpy as np

import tensorflow as tf
from kgcnn.graph.adj import get_angle_indices
from kgcnn.layers.geom import NodeDistanceEuclidean, EdgeAngle, NodePosition
from kgcnn.layers.conv.dimenet_conv import SphericalBasisLayer
from kgcnn.layers.geom import BesselBasisLayer
from kgcnn.layers.modules import LazySubtract


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
        x = np.array([[-1.26981359e-02,  1.08580416e+00,  8.00099580e-03],
               [ 2.15041600e-03, -6.03131760e-03,  1.97612040e-03],
               [ 1.01173084e+00,  1.46375116e+00,  2.76574800e-04],
               [-5.40815069e-01,  1.44752661e+00, -8.76643715e-01],
               [-5.23813634e-01,  1.43793264e+00,  9.06397294e-01]])

        ei, _, a = get_angle_indices(ei)
        ei1, _, a1 = get_angle_indices(ei1)

        rag_x = tf.RaggedTensor.from_row_lengths(np.concatenate([x,x1]), np.array([len(x),len(x1)]))
        rag_ei = tf.RaggedTensor.from_row_lengths(np.concatenate([ei,ei1]), np.array([len(ei),len(ei1)]))
        rag_a = tf.RaggedTensor.from_row_lengths(np.concatenate([a,a1]), np.array([len(a),len(a1)]))

        a, b = NodePosition()([rag_x, rag_ei])
        dist = NodeDistanceEuclidean()([a, b])
        vec = LazySubtract()([a, b])
        angs = EdgeAngle()([vec, rag_a])
        bessel = SphericalBasisLayer(10,10,5.0)([dist, angs, rag_a])

        loaded_reference = np.load("bessel_basis_reference.npz", allow_pickle=True)
        test = np.array(loaded_reference["spherical_basis_0"])
        test1 = np.array(loaded_reference["spherical_basis_1"])
        pred = np.array(bessel[0])
        pred1 = np.array(bessel[1])

        valid = np.abs(test - pred)
        valid1 = np.abs(test1 - pred1)
        # print(np.max(valid))
        # print(np.max(valid1))

        # For small argument and large order there are some rof errors
        # So we only have < 0.1 here
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
        a, b = NodePosition()([rag_x, rag_ei])
        dist = NodeDistanceEuclidean()([a, b])
        bessel = BesselBasisLayer(10,5.0)(dist)

        loaded_reference = np.load("bessel_basis_reference.npz", allow_pickle=True)
        test0 = np.array(loaded_reference["bessel_basis_0"])
        test1 = np.array(loaded_reference["bessel_basis_1"])

        # print(np.max(np.abs(test0 - bessel[0])))
        # print(np.max(np.abs(test1 - bessel[1])))
        self.assertTrue(np.max(np.abs(test0 - bessel[0])) < 1e-5)
        self.assertTrue(np.max(np.abs(test1 - bessel[1])) < 1e-5)


if __name__ == '__main__':
    unittest.main()
