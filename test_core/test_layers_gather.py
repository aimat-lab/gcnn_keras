import numpy as np

from keras_core import ops
from keras_core import testing
from kgcnn.layers_core.gather import GatherNodes


class GatherNodesTest(testing.TestCase):

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

    def test_basics(self):

        self.run_layer_test(
            GatherNodes,
            init_kwargs={
            },
            input_dtype="int32",
            input_shape=[(4, 2), (2, 8)],
            expected_output_shape=(8, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )


if __name__ == "__main__":

    GatherNodesTest().test_correctness()
    GatherNodesTest().test_basics()
    print("Tests passed.")