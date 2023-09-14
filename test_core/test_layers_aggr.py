import numpy as np
from keras_core import ops
from keras_core import testing
from kgcnn.layers_core.aggr import AggregateLocalEdges, AggregateLocalEdgesAttention


class TestAggregateLocalEdges(testing.TestCase):
    node_attr = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    edge_attr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                          [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    edge_index = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                           [0, 1, 0, 1, 2, 3, 2, 3]], dtype="int64")
    batch = np.array([0, 0, 1, 1])

    def test_correctness(self):
        layer = AggregateLocalEdges()
        nodes_aggr = layer([self.node_attr, self.edge_attr, ops.cast(self.edge_index, dtype="int64")])
        expected_output = np.array([[0., 1., 0.], [1., 1., 2.], [2., 1., 0.], [2., 1., 2.]])
        self.assertAllClose(nodes_aggr, expected_output)

    def test_basics(self):
        self.run_layer_test(
            AggregateLocalEdges,
            init_kwargs={
            },
            input_dtype="int64",
            input_shape=[(4, 2), (8, 3), (2, 8)],
            expected_output_shape=(4, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )


class TestAggregateLocalEdgesAttention(testing.TestCase):
    node_attr = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    edge_attr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                          [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    edge_att = np.array([[0.5], [2.0], [0.0], [1.0], [1.0], [1.0], [10.0], [1.0]])
    edge_index = np.array([[0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]], dtype="int64")
    batch = np.array([0, 0, 1, 1])

    def test_correctness(self):
        layer = AggregateLocalEdgesAttention()
        nodes_aggr = layer([self.node_attr, self.edge_attr, self.edge_att, ops.cast(self.edge_index, dtype="int64")])

        expected_output = [[0.0000000e+00, 0.0000000e+00, 8.1757444e-01],
                           [7.3105860e-01, 1.0000000e+00, 7.3105860e-01],
                           [1.0000000e+00, 0.0000000e+00, 5.0000000e-01],
                           [1.0000000e+00, 1.0000000e+00, 1.2339458e-04]]

        self.assertAllClose(nodes_aggr, expected_output)


if __name__ == "__main__":
    TestAggregateLocalEdges().test_correctness()
    TestAggregateLocalEdges().test_basics()
    TestAggregateLocalEdgesAttention().test_correctness()
    print("Tests passed.")
