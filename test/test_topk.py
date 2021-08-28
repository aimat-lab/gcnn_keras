import unittest

import numpy as np
import tensorflow as tf

from kgcnn.layers.pool.topk import PoolingTopK, UnPoolingTopK


class TestTopKLayerRagged(unittest.TestCase):
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
        edgeind = tf.ragged.constant(self.ei1, ragged_rank=1, inner_shape=(2,),dtype=tf.int64)
        edgefeat = tf.ragged.constant(self.e1, ragged_rank=1, inner_shape=(1,))

        out1, map1 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)([node, edgefeat,edgeind])
        out2, map2 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)(out1)
        out3, map3 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)(out2)
        out4, map4 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)(out3)
        out5, map5 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)(out4)
        out6, map6 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)(out5)
        out7, map7 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)(out6)
        out8, map8 = PoolingTopK(k=0.3, kernel_initializer="ones", ragged_validate=True)(out7)

        # Pooled to 1 node
        self.assertTrue(np.sum(np.abs(out8[0].numpy() - np.array([[[5.8759007]], [[7.9783587]]]))) < 1e-5)

        # REverse pooling
        uout7 = UnPoolingTopK(ragged_validate=True)(out7 + map8 + out8)
        uout6 = UnPoolingTopK(ragged_validate=True)(out6 + map7 + uout7)
        uout5 = UnPoolingTopK(ragged_validate=True)(out5 + map6 + uout6)
        uout4 = UnPoolingTopK(ragged_validate=True)(out4 + map5 + uout5)
        uout3 = UnPoolingTopK(ragged_validate=True)(out3 + map4 + uout4)
        uout2 = UnPoolingTopK(ragged_validate=True)(out2 + map3 + uout3)
        uout1 = UnPoolingTopK(ragged_validate=True)(out1 + map2 + uout2)
        uout = UnPoolingTopK(ragged_validate=True)([node, edgefeat, edgeind] + map1 + uout1)

        # Expected output
        unpool_nodes = [[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.8759007]],
                        [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [7.9783587], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]]

        print(uout)
        # Check unpooled
        self.assertTrue(np.sum(np.abs(uout[0][0].numpy()-np.array(unpool_nodes[0]))) < 1e-5 and np.sum(np.abs(uout[0][1].numpy()-np.array(unpool_nodes[1]))) < 1e-5)
        self.assertTrue(np.all(uout[2][1].numpy() == np.array(self.ei1[1])))
        # print(out1[0])


if __name__ == '__main__':
    unittest.main()
