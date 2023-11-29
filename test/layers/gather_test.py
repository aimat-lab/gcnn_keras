import numpy as np
from kgcnn.utils.tests import TestCase
from keras import ops
from kgcnn.layers.gather import GatherNodes


class GatherNodesTest(TestCase):

    node_attr = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    edge_attr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                          [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 1.0]])
    edge_index = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 2, 3, 2, 3]], dtype="int64")
    batch = np.array([0, 0, 1, 1])

    def test_correctness(self):

        layer = GatherNodes()
        nodes_per_edge = layer([self.node_attr, ops.cast(self.edge_index, dtype="int64")])
        expected_output = np.array([[0., 0., 0., 0., ], [0., 0., 0., 1.], [0., 1., 0., 0.], [0., 1., 0., 1.],
                                    [1., 0., 1., 0.], [1., 0., 1., 1.], [1., 1., 1., 0.], [1., 1., 1., 1.]])
        self.assertAllClose(nodes_per_edge, expected_output)


if __name__ == "__main__":

    GatherNodesTest().test_correctness()
    print("Tests passed.")