import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.modules import LazyAdd, DenseEmbedding, LazyConcatenate, LazyMultiply
from kgcnn.layers.conv.dimenet_conv import ResidualLayer
from kgcnn.layers.gather import GatherEmbeddingSelection
from kgcnn.layers.pooling import PoolingLocalMessages
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='MXMGlobalMP')
class MXMGlobalMP(GraphBaseLayer):

    def __init__(self, units: int =63, **kwargs):
        """Initialize layer."""
        super(MXMGlobalMP, self).__init__(**kwargs)
        self.dim = units
        self.h_mlp = GraphMLP(self.dim, activation="swish")
        self.res1 = ResidualLayer(self.dim)
        self.res2 = ResidualLayer(self.dim)
        self.res3 = ResidualLayer(self.dim)
        self.mlp = GraphMLP(self.dim, activation="swish")
        self.add1 = LazyAdd()

        self.x_edge_mlp = GraphMLP(self.dim, activation="swish")
        self.linear = DenseEmbedding(self.dim, bias=False, activation="linear")

        self.gather = GatherEmbeddingSelection([0,1])
        self.pool = PoolingLocalMessages()
        self.cat = LazyConcatenate()
        self.multiply_edge = LazyMultiply()
        self.add = LazyAdd()

    def build(self, input_shape):
        """Build layer."""
        super(MXMGlobalMP, self).build(input_shape)

    def propagate(self, edge_index, x, edge_attr, **kwargs):
        x_i, x_j = self.gather([x, edge_index])

        # Prepare message.
        x_edge = self.cat([x_i, x_j, edge_attr], axis=-1)
        x_edge = self.x_edge_mlp(x_edge, **kwargs)
        edge_attr_lin = self.linear(edge_attr, **kwargs)
        x_edge = self.multiply_edge([edge_attr_lin, x_edge])

        # Pooling here.
        x_p = self.pool([x, x_edge, edge_index])

        # Replace self loops by explicit node update here.
        x_i_p = self.add([x_p, x_i], dim=0)

        return x_i_p

    def call(self, inputs, **kwargs):
        h, edge_attr, edge_index = inputs

        # Keep for residual skip connections.
        res_h = h

        # Integrate the Cross Layer Mapping inside the Global Message Passing
        h = self.h_mlp(h)

        # Message Passing operation
        h = self.propagate(edge_index=edge_index, x=h, edge_attr=edge_attr, **kwargs)

        # Update function f_u
        h = self.res1(h)
        h = self.mlp(h)
        h = self.add1([h, res_h])
        h = self.res2(h)
        h = self.res3(h)

        # Message Passing operation
        h = self.propagate(edge_index=edge_index, x=h, edge_attr=edge_attr, **kwargs)

        return h


