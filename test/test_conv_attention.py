import unittest

import numpy as np
import tensorflow as tf


from kgcnn.layers.casting import ChangeTensorType,ChangeIndexing
from kgcnn.layers.conv.attention import PoolingLocalEdgesAttention, AttentionHeadGAT


class TestAttentionDisjoint(unittest.TestCase):

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

    def test_attention_pooling(self):

        layer = PoolingLocalEdgesAttention()
        result = layer([tf.RaggedTensor.from_row_lengths(np.array([[1.0],[1.0]]), np.array([1,1])),
                       tf.RaggedTensor.from_row_lengths(np.array([[100.0],[0.0],[100.0],[0.0]]),np.array([2,2])),
                       tf.RaggedTensor.from_row_lengths(np.array([[0.0],[1.0],[0.0],[1.0]]), np.array([2,2])),
                       tf.RaggedTensor.from_row_lengths(np.array([[0,1],[0,0],[1,0],[1,0]]),np.array([2,2]))]
                       )
        result = result[0].numpy()
        self.assertTrue(np.abs(result[0] - 100.0 * 1/(np.exp(1)+1) ) < 1e-4)

    def test_attention_head(self):

        n = tf.ragged.constant(self.n1, ragged_rank=1, inner_shape=(1,))
        edi = tf.ragged.constant(self.ei1, ragged_rank=1, inner_shape=(2,))
        ed = tf.ragged.constant(self.e1, ragged_rank=1, inner_shape=(1,))

        layer = AttentionHeadGAT(5)
        result = layer([n, ed, edi])

        self.assertTrue(np.all(np.array(result[0].shape) == np.array([8,5])))
        # layer.get_config()


if __name__ == '__main__':
    unittest.main()
