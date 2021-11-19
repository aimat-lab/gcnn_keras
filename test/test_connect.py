import numpy as np
import tensorflow as tf
import unittest

from kgcnn.layers.pool.topk import AdjacencyPower

nods = tf.ragged.constant([[[5.0],[5.0],[5.0]],[[5.0],[5.0],[5.0]]],ragged_rank=1)
inds = tf.ragged.constant([[[0,1],[0,2],[1,0],[2,0]],[[0,0],[1,1],[2,2]]],ragged_rank=1)
edges = tf.ragged.constant([[[1.0],[1.0],[1.0],[1.0]],[[1.0],[1.0],[1.0]]],ragged_rank=1)

edge_index = inds.values
edge = edges.values
edge_len = inds.row_lengths()
node_len = nods.row_lengths()



class TestAdjacencyPower(unittest.TestCase):

    nods = tf.ragged.constant([[[5.0], [5.0], [5.0]], [[5.0], [5.0], [5.0]]], ragged_rank=1)
    inds = tf.ragged.constant([[[0, 1], [0, 2], [1, 0], [2, 0]], [[0, 0], [1, 1], [2, 2]]], ragged_rank=1)
    edges = tf.ragged.constant([[[1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]]], ragged_rank=1)

    def test_attention_pooling(self):

        result = AdjacencyPower(input_tensor_type="ragged")([self.nods,self.edges,self.inds])
        print(result)

        self.assertTrue(np.max(np.abs(result[1][0].numpy() -  np.array([[0, 0], [1, 1], [1, 2], [2, 1], [2, 2]]))) < 1e-4)




if __name__ == '__main__':
    unittest.main()
