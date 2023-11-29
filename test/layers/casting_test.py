import numpy as np
from keras import ops
from kgcnn.utils.tests import TestCase
from kgcnn.layers.casting import CastBatchedIndicesToDisjoint, CastBatchedAttributesToDisjoint
from kgcnn.utils.tests import compare_static_shapes


class TestCastBatchedIndicesToDisjoint(TestCase):

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

        layer = CastBatchedIndicesToDisjoint(reverse_indices=True)
        layer_input = [self.nodes, ops.cast(self.edge_indices, dtype="int64"), self.node_len, self.edge_len]
        node_attr, edge_index, batch_node, batch_edge, node_id, edge_id, node_count, edge_count = layer(layer_input)
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(edge_index, [[0, 1, 2, 1], [0, 1, 1, 2]])
        self.assertAllClose(batch_node, [0, 1, 1])
        self.assertAllClose(batch_edge, [0, 1, 1, 1])
        self.assertAllClose(node_id, [0, 0, 1])
        self.assertAllClose(edge_id, [0, 0, 1, 2])
        self.assertAllClose(node_count, [1, 2])
        self.assertAllClose(edge_count, [1, 3])

        output_shape = layer.compute_output_shape([x.shape for x in layer_input])
        expected_output_shape = [(None, 2), (2, None), (None, ), (None, ), (None, ), (None, ), (None, ), (None, )]
        for f, e in zip(output_shape, expected_output_shape):
            self.assertTrue(compare_static_shapes(f, e), msg=f"Shape mismatch: {f} vs. {e}")

    def test_correctness_padding(self):

        layer = CastBatchedIndicesToDisjoint(padded_disjoint=True, reverse_indices=True)
        layer_input = [self.nodes, ops.cast(self.edge_indices, dtype="int64"), self.node_len, self.edge_len]
        node_attr, edge_index, batch_node, batch_edge, node_id, edge_id, node_count, edge_count = layer(layer_input)

        self.assertAllClose(node_attr, [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(edge_index, [[0, 1, 0, 0, 0, 3, 4, 3, 0], [0, 1, 0, 0, 0, 3, 3, 4, 0]])
        self.assertAllClose(batch_node, [0, 1, 0, 2, 2])
        self.assertAllClose(batch_edge, [0, 1, 0, 0, 0, 2, 2, 2, 0])
        self.assertAllClose(node_id, [0, 0, 0, 0, 1])
        self.assertAllClose(edge_id, [0, 0, 0, 0, 0, 0, 1, 2, 0])
        self.assertAllClose(node_count, [1, 1, 2])
        self.assertAllClose(edge_count, [4, 1, 3])

        output_shape = layer.compute_output_shape([x.shape for x in layer_input])
        expected_output_shape = [(5, 2), (2, 9), (5, ), (9, ), (5, ), (9, ), (3, ), (3, )]
        for f, e in zip(output_shape, expected_output_shape):
            self.assertTrue(compare_static_shapes(f, e), msg=f"Shape mismatch: {f} vs. {e}")


class TestCastBatchedAttributesToDisjoint(TestCase):

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

        layer = CastBatchedAttributesToDisjoint(reverse_indices=True)
        layer_input = [self.nodes, self.node_len]
        node_attr, graph_id_node, node_id, node_len = layer(layer_input)
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(graph_id_node, [0, 1, 1])
        self.assertAllClose(node_id, [0, 0, 1])
        self.assertAllClose(node_len, [1, 2])

        output_shape = layer.compute_output_shape([x.shape for x in layer_input])
        expected_output_shape = [(None, 2), (None, ), (None, ), (2, )]
        for f, e in zip(output_shape, expected_output_shape):
            self.assertTrue(compare_static_shapes(f, e), msg=f"Shape mismatch: {f} vs. {e}")

        layer = CastBatchedAttributesToDisjoint(reverse_indices=True)
        layer_input = [self.edges, self.edge_len]
        edge_attr, graph_id_edge, edge_id, edge_len = layer(layer_input)
        self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertAllClose(graph_id_edge, [0, 1, 1, 1])
        self.assertAllClose(edge_id, [0, 0, 1, 2])
        self.assertAllClose(edge_len, [1, 3])

        output_shape = layer.compute_output_shape([x.shape for x in layer_input])
        expected_output_shape = [(None, 3), (None, ), (None, ), (2, )]
        for f, e in zip(output_shape, expected_output_shape):
            self.assertTrue(compare_static_shapes(f, e), msg=f"Shape mismatch: {f} vs. {e}")

    def test_correctness_padding(self):

        layer = CastBatchedAttributesToDisjoint(padded_disjoint=True, reverse_indices=True)
        layer_input = [self.nodes, self.node_len]
        node_attr, graph_id_node, node_id, node_len = layer(layer_input)
        self.assertAllClose(node_attr, [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(graph_id_node, [0, 1, 0, 2, 2])
        self.assertAllClose(node_id, [0, 0, 0, 0, 1])
        self.assertAllClose(node_len, [1, 1, 2])

        output_shape = layer.compute_output_shape([x.shape for x in layer_input])
        expected_output_shape = [(5, 2), (5, ), (5, ), (3, )]
        for f, e in zip(output_shape, expected_output_shape):
            self.assertTrue(compare_static_shapes(f, e), msg=f"Shape mismatch: {f} vs. {e}")

        layer = CastBatchedAttributesToDisjoint(padded_disjoint=True)
        layer_input = [self.edges, self.edge_len]
        edge_attr, graph_id_edge, edge_id, edge_len = layer(layer_input)
        self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 1.0]])
        self.assertAllClose(graph_id_edge, [0, 1, 0, 0, 0, 2, 2, 2, 0])
        self.assertAllClose(edge_id, [0, 0, 0, 0, 0, 0, 1, 2, 0])
        self.assertAllClose(edge_len, [4, 1, 3])

        output_shape = layer.compute_output_shape([x.shape for x in layer_input])
        expected_output_shape = [(9, 3), (9, ), (9, ), (3, )]
        for f, e in zip(output_shape, expected_output_shape):
            self.assertTrue(compare_static_shapes(f, e), msg=f"Shape mismatch: {f} vs. {e}")


if __name__ == "__main__":

    TestCastBatchedIndicesToDisjoint().test_correctness()
    TestCastBatchedIndicesToDisjoint().test_correctness_padding()
    TestCastBatchedAttributesToDisjoint().test_correctness()
    TestCastBatchedAttributesToDisjoint().test_correctness_padding()
    print("Tests passed.")
