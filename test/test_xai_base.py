import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.xai.testing import MockContext
from kgcnn.xai.base import MockImportanceExplanationMethod


# == UNIT TESTS ==

class TestMockImportanceExplanationMethod(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock = MockContext()
        cls.mock.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock.__exit__(None, None, None)

    def test_basically_works(self):
        channels = 3
        xai_instance = MockImportanceExplanationMethod(channels=channels)
        node_importances, edge_importances = xai_instance(
            self.mock.model,
            self.mock.x,
            self.mock.y
        )
        assert isinstance(node_importances, tf.RaggedTensor)
        assert isinstance(edge_importances, tf.RaggedTensor)

        node_importances = node_importances.numpy()
        edge_importances = edge_importances.numpy()
        for v, w in zip(node_importances, edge_importances):
            assert v.shape[-1] == channels
            assert w.shape[-1] == channels
