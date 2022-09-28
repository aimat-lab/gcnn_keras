import unittest

import os
import random
import itertools
import tempfile

import numpy as np
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
            tf.ragged.constant(n, ragged_rank=1, dtype=tf.float32),
            tf.ragged.constant(e, ragged_rank=1, dtype=tf.float32),
            tf.ragged.constant(ei, ragged_rank=1, dtype=tf.int32),
        )

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

    def test_saving_loading_basically_works(self):
        num_batches = 5
        num_features = 3
        n, e, eid = self.random_input(num_batches=num_batches, num_features=num_features)

        num_channels = 2
        num_out = 1
        model = MEGAN(
            units=[5, 3],
            importance_channels=num_channels,
            final_units=[num_out]
        )
        # At this point the model is not built yet and should raise a value error
        with self.assertRaises(ValueError):
            model.summary()

        out, ni, ei = model([n, e, eid], training=False)
        self.assertEqual((num_batches, num_out), out.shape)
        self.assertEqual((num_batches, None, num_channels), ni.shape)
        self.assertEqual((num_batches, None, num_channels), ei.shape)

        # After having passed in some input, the model should be built and that should work
        model.summary()
        weights = model.get_weights()

        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'model')

            # Saving the model to a file
            model.save(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Loading the model from that file again
            model_loaded = ks.models.load_model(
                model_path,
                custom_objects={'MEGAN': MEGAN}
            )
            self.assertIsInstance(model_loaded, ks.models.Model)

            out_l, ni_l, ei_l = model_loaded([n, e, eid], training=False)
            # Basic test for correct shape
            self.assertEqual((num_batches, num_out), out_l.shape)
            self.assertEqual((num_batches, None, num_channels), ni_l.shape)
            self.assertEqual((num_batches, None, num_channels), ei_l.shape)

            # Comparing the model weights
            weights_l = model_loaded.get_weights()
            for w, w_l in zip(weights, weights_l):
                np.testing.assert_allclose(w, w_l)

            # Testing if both models produce the same outputs, given the same inputs
            np.testing.assert_allclose(out, out_l, rtol=1e-3)
            np.testing.assert_allclose(ni.values, ni_l.values, rtol=1e-3)
            np.testing.assert_allclose(ei.values, ei_l.values, rtol=1e-3)

    def test_explanation_only_training_basically_works(self):
        num_batches = 20
        num_features = 3
        n, e, eid = self.random_input(num_batches=num_batches, num_features=num_features)
        out = tf.ragged.constant([random.random() for _ in range(num_batches)])

        num_channels = 2
        num_out = 1
        model = MEGAN(
            units=[5, 3],
            importance_channels=num_channels,
            final_units=[num_out],
            return_importances=False,
            # We explicitly want to try using the explanation step here
            importance_factor=1.0,
            # We need this to indicate that we are trying to do regression here:
            regression_limits=(-1, 1),
            regression_reference=0
        )

        model.compile(
            optimizer='adam',
            loss=ks.losses.mean_squared_error
        )
        history = model.fit(
            [n, e, eid], out,
            batch_size=1,
            epochs=2,
            verbose=0
        )
        self.assertIn('exp_loss', history.history)
        self.assertNotEqual(0.0, history.history['exp_loss'])

