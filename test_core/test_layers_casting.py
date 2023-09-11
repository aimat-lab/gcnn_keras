import numpy as np

from keras_core import ops
from keras_core import testing
from kgcnn.layers_core.casting import CastBatchedGraphIndicesToPyGDisjoint, CastBatchedGraphAttributesToPyGDisjoint


class CastBatchedGraphsToPyGDisjointTest(testing.TestCase):

    nodes = np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
    edges = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                      [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 1.0]]])
    edge_indices = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
                             [[0, 0], [0, 1], [1, 0], [1, 1]]], dtype="int64")
    node_mask = np.array([[True, False], [True, True]])
    edge_mask = np.array([[True, False, False, False], [True, True, True, False]])
    node_len = np.array([1, 2], dtype="int64")
    edge_len = np.array([1, 3], dtype="int64")

    def test_correctness(self):

        layer = CastBatchedGraphIndicesToPyGDisjoint()
        node_attr, edge_index, batch, _, _ = layer(
            [self.nodes, ops.cast(self.edge_indices, dtype="int64"),
             self.node_len, self.edge_len
             ])
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        # self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertAllClose(edge_index, [[0, 1, 2, 1], [0, 1, 1, 2]])
        self.assertAllClose(batch, [0, 1, 1])


class TestCastBatchedGraphAttributesToPyGDisjoint(testing.TestCase):

    nodes = np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
    edges = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                      [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 1.0]]])
    edge_indices = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
                             [[0, 0], [0, 1], [1, 0], [1, 1]]], dtype="int64")
    node_mask = np.array([[True, False], [True, True]])
    edge_mask = np.array([[True, False, False, False], [True, True, True, False]])
    node_len = np.array([1, 2], dtype="int64")
    edge_len = np.array([1, 3], dtype="int64")

    def test_correctness(self):

        layer = CastBatchedGraphAttributesToPyGDisjoint()
        node_attr, _ = layer(
            [self.nodes, self.node_len])
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])



if __name__ == "__main__":
    CastBatchedGraphsToPyGDisjointTest().test_correctness()
    TestCastBatchedGraphAttributesToPyGDisjoint().test_correctness()
    print("Tests passed.")
