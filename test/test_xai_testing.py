import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.xai.testing import MockContext


class TestMockContext(unittest.TestCase):

    def test_basically_works(self):
        num_elements = 10
        num_targets = 2
        with MockContext(num_elements=num_elements, num_targets=num_targets) as mock:
            assert isinstance(mock, MockContext)

            assert isinstance(mock.model, ks.models.Model)
            assert mock.model.built

            assert isinstance(mock.x, tuple)
            assert len(mock.x) == 3
            for element in mock.x:
                print(element.shape)
                assert isinstance(element, tf.RaggedTensor)

            assert isinstance(mock.y, tuple)
            assert len(mock.y) == 1

            targets = mock.y[0]
            assert isinstance(targets, np.ndarray)
            assert len(targets.shape) == 2
            assert targets.shape[-1] == num_targets
