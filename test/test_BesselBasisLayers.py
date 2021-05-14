import tensorflow as tf
import numpy as np
from kgcnn.layers.geom import SphericalBasisLayer, NodeDistance, EdgeAngle

spb = SphericalBasisLayer(3,3,2.0)
dist = NodeDistance()
angs = EdgeAngle()

x = tf.RaggedTensor.from_row_lengths(np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0],[1,0,0]], dtype=np.float32), np.array([3,3],dtype=np.int64))
edi = tf.RaggedTensor.from_row_lengths(np.array([[0,1],[0,2],[1,0],[2,0],[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]],dtype=np.int64), np.array([4,6],dtype=np.int64))
adi = tf.RaggedTensor.from_row_lengths(np.array([[0,1],[0,3],[1,0],[1,2],[2,1],[2,3],[3,0],[3,2],
                                                 [0,1],[0,3],[0,4],[0,5],
                                                 [1,0],[1,2],[1,3],[1,5],
                                                 [2,1],[2,3],[2,4],[2,5],
                                                 [3,0],[3,1],[3,2],[3,4],
                                                 [4,0],[4,2],[4,3],[4,5],
                                                 [5,0],[5,1],[5,2],[5,4],
                                                 ],dtype=np.int64), np.array([8,24],dtype=np.int64))

d = dist([x,edi])
a = angs([x, edi, adi])
basis = spb([d,a,adi])
