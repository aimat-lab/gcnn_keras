import unittest

import numpy as np
import tensorflow as tf
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.layers.conv.dmpnn_conv import DMPNNGatherEdgesPairs

class TestReverseEdges(unittest.TestCase):

    n1 = [[[2.0],[3.0],[4.0]], [[1.0],[10.0],[100.0]]]
    ei1 = [[[0, 1], [1, 0], [1, 2], [2,1]],[[0,1],[1,2],[2,1],[2,0]]]
    e1 = [[[0.0,0.0],[1.0,1.0],[2.0,2.0],[3.0,3.0],[4.0,4.0]],
          [[0.0,0.0],[1.0,1.0],[2.0,2.0],[3.0,3.0]]]

    def test_gather(self):
        edge = tf.ragged.constant(self.e1, ragged_rank=1, inner_shape=(2,))
        edgeind = tf.ragged.constant(self.ei1, ragged_rank=1, inner_shape=(2,))
        ds = MemoryGraphDataset()
        ds.edge_indices = [np.array(self.ei1[0]), np.array(self.ei1[1])]
        ds.set_edge_indices_reverse()
        edge_pair = tf.RaggedTensor.from_row_lengths(np.concatenate(ds.edge_indices_reverse, axis=0), [len(x) for x in ds.edge_indices_reverse])
        edges_gather = DMPNNGatherEdgesPairs()([edge, edge_pair])
        result = edges_gather

        self.assertTrue(np.amax(np.abs(np.array(result[0]) - np.array([[1.0, 1.0], [0.0, 0.0], [3.0, 3.0], [2.0, 2.0]]))) < 1e-4)

        # layer.get_config()


if __name__ == '__main__':
    unittest.main()
