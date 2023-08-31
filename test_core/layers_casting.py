import numpy as np

from keras_core import ops
from keras_core import testing
from kgcnn.layers_core.casting import CastBatchGraphListToPyGDisjoint


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

    def test_correctness_lengths(self):

        layer = CastBatchGraphListToPyGDisjoint()
        node_attr, edge_attr, edge_index, batch = layer(
            [self.nodes, self.edges,
             ops.cast(self.edge_indices, dtype="int64"),
             self.node_len, self.edge_len
             ])
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertAllClose(edge_index, [[0, 1, 2, 1], [0, 1, 1, 2]])
        self.assertAllClose(batch, [0, 1, 1])

    def test_correctness_mask(self):

        layer = CastBatchGraphListToPyGDisjoint(reverse_indices=False, batch_info="mask")
        node_attr, edge_attr, edge_index, batch = layer(
            [self.nodes, self.edges, ops.cast(self.edge_indices, dtype="int64"), self.node_mask, self.edge_mask],
        )
        self.assertAllClose(node_attr, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertAllClose(edge_index, [[0, 1, 1, 2], [0, 1, 2, 1]])
        self.assertAllClose(batch, [0, 1, 1])

    def test_correctness_equal_size(self):

        layer = CastBatchGraphListToPyGDisjoint(reverse_indices=False)
        node_attr, edge_attr, edge_index, batch = layer(
            [self.nodes, self.edges, ops.cast(self.edge_indices, dtype="int64")]
        )
        self.assertAllClose(node_attr, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAllClose(edge_attr, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                                        [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 1.0]])
        self.assertAllClose(edge_index, [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 2, 3, 2, 3]])
        self.assertAllClose(batch, [0, 0, 1, 1])

    def test_basics(self):
        # Normal tensor input.
        self.run_layer_test(
            CastBatchGraphListToPyGDisjoint,
            init_kwargs={
            },
            input_shape=[(2, 2, 2), (2, 4, 3), (2, 4, 2), (2, ), (2, )],
            expected_output_shape=[(4, 2), (8, 3), (2, 8), (4,)],
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )


CastBatchedGraphsToPyGDisjointTest().test_correctness_lengths()
CastBatchedGraphsToPyGDisjointTest().test_correctness_mask()
CastBatchedGraphsToPyGDisjointTest().test_correctness_equal_size()
CastBatchedGraphsToPyGDisjointTest().test_basics()
