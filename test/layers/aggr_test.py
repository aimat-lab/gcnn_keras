import numpy as np
from keras import ops
from kgcnn.utils.tests import TestCase
from kgcnn.layers.aggr import AggregateLocalEdges, AggregateLocalEdgesAttention


class TestAggregateLocalEdges(TestCase):
    node_attr = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    edge_attr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                          [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    edge_index = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                           [0, 1, 0, 1, 2, 3, 2, 3]], dtype="int64")
    batch = np.array([0, 0, 1, 1])

    def test_correctness(self):
        layer = AggregateLocalEdges(pooling_index=1)
        nodes_aggr = layer([self.node_attr, self.edge_attr, ops.cast(self.edge_index, dtype="int64")])
        expected_output = np.array([[0., 1., 0.], [1., 1., 2.], [2., 1., 0.], [2., 1., 2.]])
        self.assertAllClose(nodes_aggr, expected_output)

    def test_correctness_mean(self):
        layer = AggregateLocalEdges(pooling_method="mean", pooling_index=0)
        nodes_aggr = layer([self.node_attr, self.edge_attr, ops.cast(self.edge_index, dtype="int64")])
        expected_output = np.array([[0., 0., 0.5], [0.5, 1., 0.5], [1., 0., 0.5], [1., 1., 0.5]])
        self.assertAllClose(nodes_aggr, expected_output)


class TestAggregateLocalEdgesAttention(TestCase):
    node_attr = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    edge_attr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                          [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    edge_att = np.array([[0.5], [2.0], [0.0], [1.0], [1.0], [1.0], [10.0], [1.0]])
    edge_index = np.array([[0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]], dtype="int64")
    batch = np.array([0, 0, 1, 1])

    def test_correctness(self):
        layer = AggregateLocalEdgesAttention(pooling_index=1)
        nodes_aggr = layer([self.node_attr, self.edge_attr, self.edge_att, ops.cast(self.edge_index, dtype="int64")])

        expected_output = [[0.0000000e+00, 0.0000000e+00, 8.1757444e-01],
                           [7.3105860e-01, 1.0000000e+00, 7.3105860e-01],
                           [1.0000000e+00, 0.0000000e+00, 5.0000000e-01],
                           [1.0000000e+00, 1.0000000e+00, 1.2339458e-04]]

        self.assertAllClose(nodes_aggr, expected_output)


if __name__ == "__main__":
    TestAggregateLocalEdges().test_correctness()
    TestAggregateLocalEdges().test_correctness_mean()
    TestAggregateLocalEdgesAttention().test_correctness()
    print("Tests passed.")
