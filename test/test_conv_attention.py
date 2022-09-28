import unittest
import random

import numpy as np
import tensorflow as tf


from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.conv.attention import PoolingLocalEdgesAttention
from kgcnn.layers.conv.gat_conv import AttentionHeadGAT
from kgcnn.layers.conv.gat_conv import MultiHeadGATV2Layer


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


class TestMultiHeadGATV2Layer(unittest.TestCase):

    def random_input(self, num_batches=5, num_features=3):
        n = []
        ei = []
        e = []
        for b in range(num_batches):
            N = random.randint(5, 30)
            node_indices = list(range(N))
            M = random.randint(1, N - 1)

            n.append([[random.random() for _ in range(num_features)] for _ in range(N)])
            e.append([[random.random() for _ in range(num_features)] for _ in range(M)])
            ei.append(list(zip(random.sample(node_indices, M), random.sample(node_indices, M))))

        return (
            tf.ragged.constant(n, ragged_rank=1),
            tf.ragged.constant(e, ragged_rank=1),
            tf.ragged.constant(ei, ragged_rank=1),
        )

    def test_construction_basically_works(self):
        layer = MultiHeadGATV2Layer(units=2, num_heads=2)
        self.assertIsInstance(layer, MultiHeadGATV2Layer)
        self.assertIsInstance(layer, GraphBaseLayer)

    def test_shapes_basically_work(self):
        num_batches = 5
        n, e, ei = self.random_input(num_batches=num_batches, num_features=3)
        # print(n.shape, e.shape, ei.shape)

        # When concatenating the heads
        num_units = 2
        num_heads = 4
        layer = MultiHeadGATV2Layer(units=num_units, num_heads=num_heads, concat_heads=True)

        node_embeddings, attention_logits = layer([n, e, ei])
        self.assertIsInstance(node_embeddings, tf.RaggedTensor)
        self.assertIsInstance(attention_logits, tf.RaggedTensor)
        self.assertEqual((num_batches, None, num_units * num_heads), node_embeddings.shape)
        self.assertEqual((num_batches, None, num_heads, 1), attention_logits.shape)

        # When not concatenating the heads -> should average instead
        layer = MultiHeadGATV2Layer(units=num_units, num_heads=num_heads, concat_heads=False)
        node_embeddings, attention_logits = layer([n, e, ei])
        self.assertEqual((num_batches, None, num_units), node_embeddings.shape)
        self.assertEqual((num_batches, None, num_heads, 1), attention_logits.shape)



if __name__ == '__main__':
    unittest.main()
