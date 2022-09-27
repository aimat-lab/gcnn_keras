import unittest
import random
import itertools

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.literature.MEGAN import MEGAN


class TestMegan(unittest.TestCase):

    def random_input(self, num_batches, num_features):
        n = []
        ei = []
        e = []
        for b in range(num_batches):
            N = random.randint(5, 30)
            node_indices = list(range(N))
            M = random.randint(1, N - 1)

            n.append([[random.random() for _ in range(num_features)] for _ in range(N)])
            e.append([[random.random() for _ in range(num_features)] for _ in range(M)])
            ei.append(list(zip(random.sample(node_indices, M), random.sample(node_indices, M))))

        return (
            tf.ragged.constant(n, ragged_rank=1),
            tf.ragged.constant(e, ragged_rank=1),
            tf.ragged.constant(ei, ragged_rank=1),
        )

    def test_ragged_tensor_from_shape(self):
        tensor = self.ragged_tensor_from_shape((None, 3), 'float32')


    # -- UNITTESTS --

    def test_construction_basically_works(self):
        model = MEGAN(units=[1])
        self.assertIsInstance(model, MEGAN)
        self.assertIsInstance(model, ks.models.Model)

    def test_shapes_basically_work(self):
        num_batches = 5
        num_features = 3
        n, e, ei = self.random_input(num_batches=num_batches, num_features=num_features)

        # We just test a bunch of different configurations for the output dimension and the number of
        # explanation channels.
        output_dimensions = [1, 3]
        channel_dimensions = [2, 6]

        for num_out, num_channels in itertools.product(output_dimensions, channel_dimensions):
            model = MEGAN(
                units=[3],
                importance_channels=num_channels,
                final_units=[num_out],
            )

            out, node_importances, edge_importances = model([n, e, ei])
            self.assertEqual((num_batches, num_out), out.shape)
            self.assertEqual((num_batches, None, num_channels), node_importances.shape)
            self.assertEqual((num_batches, None, num_channels), node_importances.shape)
