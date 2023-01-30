import unittest

import tensorflow as tf

from kgcnn.xai.testing import MockContext
from kgcnn.literature.GNNExplain import GnnExplainer


class TestGnnExplainer(unittest.TestCase):

    def test_basically_works(self):
        num_targets = 1
        with MockContext(num_targets=num_targets) as mock:
            gnn_explainer = GnnExplainer(
                channels=num_targets,
                verbose=True
            )
            node_importances, edge_importances = gnn_explainer(
                model=mock.model,
                x=mock.x,
                y=mock.y,
            )
            assert isinstance(node_importances, tf.RaggedTensor)
            assert isinstance(edge_importances, tf.RaggedTensor)

    def test_multiple_targets_works(self):
        num_targets = 2
        with MockContext(num_targets=num_targets) as mock:
            gnn_explainer = GnnExplainer(
                channels=num_targets,
                verbose=True
            )
            node_importances, edge_importances = gnn_explainer(
                model=mock.model,
                x=mock.x,
                y=mock.y,
            )
            assert isinstance(node_importances, tf.RaggedTensor)
            assert isinstance(edge_importances, tf.RaggedTensor)


