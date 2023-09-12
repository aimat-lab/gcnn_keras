import numpy as np

from keras_core import ops
from keras_core import testing
from kgcnn.layers_core.casting import CastBatchedGraphIndicesToDisjoint, CastBatchedGraphAttributesToDisjoint


class CastBatchedGraphsToDisjointTest(testing.TestCase):

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

        layer = CastBatchedGraphIndicesToDisjoint()
        node_attr, edge_index, node_count, _, _, _, edge_attr = layer(
            [self.nodes, ops.cast(self.edge_indices, dtype="int64"),
             self.node_len, self.edge_len, self.edges
             ])
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertAllClose(edge_index, [[0, 1, 2, 1], [0, 1, 1, 2]])
        self.assertAllClose(node_count, [0, 1, 1])

    def test_correctness_padding(self):

        layer = CastBatchedGraphIndicesToDisjoint(padded_disjoint=True)
        node_attr, edge_index, batch_node, batch_edge, node_count, edge_count, edge_attr = layer(
            [self.nodes, ops.cast(self.edge_indices, dtype="int64"),
             self.node_len, self.edge_len, self.edges
             ])
        self.assertAllClose(node_attr, [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 1.0]])
        self.assertAllClose(edge_index, [[0, 1, 0, 0, 0, 3, 4, 3, 0], [0, 1, 0, 0, 0, 3, 3, 4, 0]])
        self.assertAllClose(node_count, [1, 2, 2])


class TestCastBatchedGraphAttributesToDisjoint(testing.TestCase):

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

        layer = CastBatchedGraphAttributesToDisjoint()
        node_attr, _, _ = layer(
            [self.nodes, self.node_len])
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])


if __name__ == "__main__":

    CastBatchedGraphsToDisjointTest().test_correctness()
    CastBatchedGraphsToDisjointTest().test_correctness_padding()
    TestCastBatchedGraphAttributesToDisjoint().test_correctness()
    print("Tests passed.")
