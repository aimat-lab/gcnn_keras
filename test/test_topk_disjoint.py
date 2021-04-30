import unittest

import numpy as np
import tensorflow as tf

from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing
from kgcnn.layers.topk import PoolingTopK, UnPoolingTopK


class TestTopKLayerDisjoint(unittest.TestCase):
    n1 = [[[1.0], [6.0], [1.0], [6.0], [1.0], [1.0], [6.0], [6.0]],
          [[6.0], [1.0], [1.0], [1.0], [7.0], [1.0], [6.0], [8.0], [6.0], [1.0], [6.0], [7.0], [1.0], [1.0], [1.0]]]
    ei1 = [[[0, 1], [1, 0], [1, 6], [2, 3], [3, 2], [3, 5], [3, 7], [4, 7], [5, 3], [6, 1], [6, 7], [7, 3], [7, 4],
            [7, 6]],
           [[0, 6], [0, 8], [0, 9], [1, 11], [2, 4], [3, 4], [4, 2], [4, 3], [4, 6], [5, 10], [6, 0], [6, 4], [6, 14],
            [7, 8], [8, 0], [8, 7], [8, 11], [9, 0], [10, 5], [10, 11], [10, 12], [10, 13], [11, 1], [11, 8], [11, 10],
            [12, 10], [13, 10], [14, 6]]]
    e1 = [[[0.408248290463863], [0.408248290463863], [0.3333333333333334], [0.35355339059327373], [0.35355339059327373],
           [0.35355339059327373], [0.25], [0.35355339059327373], [0.35355339059327373], [0.3333333333333334],
           [0.2886751345948129], [0.25], [0.35355339059327373], [0.2886751345948129]],
          [[0.25], [0.25], [0.35355339059327373], [0.35355339059327373], [0.35355339059327373], [0.35355339059327373],
           [0.35355339059327373], [0.35355339059327373], [0.25], [0.3162277660168379], [0.25], [0.25],
           [0.35355339059327373], [0.35355339059327373], [0.25], [0.35355339059327373], [0.25], [0.35355339059327373],
           [0.3162277660168379], [0.22360679774997896], [0.3162277660168379], [0.3162277660168379],
           [0.35355339059327373], [0.25], [0.22360679774997896], [0.3162277660168379], [0.3162277660168379],
           [0.35355339059327373]]]

    def test_pool_multiple_times(self):
        node = tf.ragged.constant(self.n1, ragged_rank=1, inner_shape=(1,))
        edgeind = tf.ragged.constant(self.ei1, ragged_rank=1, inner_shape=(2,))
        edgefeat = tf.ragged.constant(self.e1, ragged_rank=1, inner_shape=(1,))

        node_indexing = 'sample'
        partition_type = 'row_length'
        tens_type = "values_partition"
        n = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(node)
        ed = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(edgefeat)
        edi = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(edgeind)
        edi = ChangeIndexing(input_tensor_type=tens_type, to_indexing=node_indexing)([n, edi])
        dislist =  [n, ed, edi]

        pool_dislist, pool_map = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                             partition_type=partition_type, input_tensor_type=tens_type)(dislist)
        pool_dislist2, pool_map2 = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                               partition_type=partition_type, input_tensor_type=tens_type)(pool_dislist)
        pool_dislist3, pool_map3 = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                               partition_type=partition_type, input_tensor_type=tens_type)(pool_dislist2)
        pool_dislist4, pool_map4 = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                               partition_type=partition_type, input_tensor_type=tens_type)(pool_dislist3)
        pool_dislist5, pool_map5 = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                               partition_type=partition_type, input_tensor_type=tens_type)(pool_dislist4)
        pool_dislist6, pool_map6 = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                               partition_type=partition_type, input_tensor_type=tens_type)(pool_dislist5)
        pool_dislist7, pool_map7 = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                               partition_type=partition_type, input_tensor_type=tens_type)(pool_dislist6)
        pool_dislist8, pool_map8 = PoolingTopK(k=0.3, kernel_initializer="ones", node_indexing=node_indexing,
                                               partition_type=partition_type, input_tensor_type=tens_type)(pool_dislist7)

        # Pooled to 1 node
        self.assertTrue(np.sum(np.abs(pool_dislist8[0][0].numpy() - np.array([[5.8759007], [7.9783587]]))) < 1e-5)
        self.assertTrue(np.all(pool_dislist8[1][0].numpy() == np.array([1, 1])))
        # print(pool_map[0])

        # Unpooling
        unpool_dislist7 = UnPoolingTopK(node_indexing=node_indexing, partition_type=partition_type, input_tensor_type=tens_type)(
            pool_dislist7 + pool_map8 + pool_dislist8)
        unpool_dislist6 = UnPoolingTopK(node_indexing=node_indexing, partition_type=partition_type, input_tensor_type=tens_type)(
            pool_dislist6 + pool_map7 + unpool_dislist7)
        unpool_dislist5 = UnPoolingTopK(node_indexing=node_indexing, partition_type=partition_type, input_tensor_type=tens_type)(
            pool_dislist5 + pool_map6 + unpool_dislist6)
        unpool_dislist4 = UnPoolingTopK(node_indexing=node_indexing, partition_type=partition_type, input_tensor_type=tens_type)(
            pool_dislist4 + pool_map5 + unpool_dislist5)
        unpool_dislist3 = UnPoolingTopK(node_indexing=node_indexing, partition_type=partition_type, input_tensor_type=tens_type)(
            pool_dislist3 + pool_map4 + unpool_dislist4)
        unpool_dislist2 = UnPoolingTopK(node_indexing=node_indexing, partition_type=partition_type, input_tensor_type=tens_type)(
            pool_dislist2 + pool_map3 + unpool_dislist3)
        unpool_dislist = UnPoolingTopK(node_indexing=node_indexing, partition_type=partition_type, input_tensor_type=tens_type)(
            pool_dislist + pool_map2 + unpool_dislist2)
        dislist_new = UnPoolingTopK(node_indexing=node_indexing,partition_type=partition_type, input_tensor_type=tens_type)(
            list(dislist) + pool_map + unpool_dislist)

        # Expected output
        unpool_nodes = np.array([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.8759007],
                                 [0.], [0.], [0.], [0.], [0.], [0.], [0.], [7.9783587], [0.], [0.], [0.], [0.], [0.],
                                 [0.], [0.]])

        # Check unpooled
        self.assertTrue(np.sum(np.abs(dislist_new[0][0]-unpool_nodes)) < 1e-5)
        self.assertTrue(np.all(dislist_new[2][0].numpy() == dislist[2][0].numpy()))




if __name__ == '__main__':
    unittest.main()
