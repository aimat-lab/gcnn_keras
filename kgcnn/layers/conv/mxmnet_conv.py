import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.modules import LazyAdd, DenseEmbedding, LazyConcatenate, LazyMultiply
from kgcnn.layers.conv.dimenet_conv import ResidualLayer
from kgcnn.layers.gather import GatherEmbeddingSelection, GatherNodesOutgoing
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
        self.add_res = LazyAdd()

        self.x_edge_mlp = GraphMLP(self.dim, activation="swish")
        self.linear = DenseEmbedding(self.dim, use_bias=False, activation="linear")

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
        x_i_p = self.add([x_p, x])

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
        h = self.add_res([h, res_h])
        h = self.res2(h)
        h = self.res3(h)

        # Message Passing operation
        h = self.propagate(edge_index=edge_index, x=h, edge_attr=edge_attr, **kwargs)

        return h


class MXMLocalMP(GraphBaseLayer):

    def __init__(self, units, **kwargs):
        super(MXMLocalMP, self).__init__(**kwargs)
        self.dim = units

        self.h_mlp = GraphMLP(self.dim)

        self.mlp_kj = GraphMLP([self.dim], activation="swish")
        self.mlp_ji_1 = GraphMLP([self.dim], activation="swish")
        self.mlp_ji_2 = GraphMLP([self.dim], activation="swish")
        self.mlp_jj = GraphMLP([self.dim], activation="swish")

        self.mlp_sbf1 = GraphMLP([self.dim, self.dim], activation="swish")
        self.mlp_sbf2 = GraphMLP([self.dim, self.dim], activation="swish")
        self.lin_rbf1 = DenseEmbedding(self.dim, use_bias=False, activation="linear")
        self.lin_rbf2 = DenseEmbedding(self.dim, use_bias=False, activation="linear")

        self.res1 = ResidualLayer(self.dim)
        self.res2 = ResidualLayer(self.dim)
        self.res3 = ResidualLayer(self.dim)

        self.lin_rbf_out = DenseEmbedding(self.dim, use_bias=False, activation="linear")

        self.h_mlp = GraphMLP([self.dim], activation="swish")

        self.y_mlp = GraphMLP([self.dim, self.dim, self.dim], activation="swish")
        self.y_W = DenseEmbedding(1)
        self.add_res = LazyAdd()

        self.gather_nodes = GatherEmbeddingSelection([0, 1])
        self.cat = LazyConcatenate()
        self.multiply = LazyMultiply()
        self.gather_mkj = GatherNodesOutgoing()
        self.gather_mjj = GatherNodesOutgoing()
        self.pool_mkj = PoolingLocalMessages(pooling_method="sum")
        self.pool_mjj = PoolingLocalMessages(pooling_method="sum")
        self.pool_h = PoolingLocalMessages(pooling_method="sum")
        self.add_mji_1 = LazyAdd()
        self.add_mji_2 = LazyAdd()

    def call(self, inputs, **kwargs):
        h, rbf, sbf1, sbf2, edge_index, angle_idx_1, angle_idx_2 = inputs
        res_h = h

        # Integrate the Cross Layer Mapping inside the Local Message Passing
        h = self.h_mlp(h, **kwargs)

        # Message Passing 1
        hi, hj = self.gather_nodes([h, edge_index])
        m = self.cat([hi, hj, rbf])

        m_kj = self.mlp_kj(m, **kwargs)
        m_kj = self.multiply([m_kj, self.lin_rbf1(rbf)])
        m_kj = self.gather_mkj([m_kj, angle_idx_1])
        m_kj = self.multiply([m_kj, self.mlp_sbf1(sbf1)])
        m_kj = self.pool_mkj([m, m_kj, angle_idx_1])

        m_ji_1 = self.mlp_ji_1(m, **kwargs)

        m = self.add_mji_1([m_ji_1, m_kj])

        # Message Passing 2       (index jj denotes j'i in the main paper)
        m_jj = self.mlp_jj(m, **kwargs)
        m_jj = self.multiply([m_jj, self.lin_rbf2(rbf)])
        m_jj = self.gather_mjj([m_jj, angle_idx_2])
        m_jj = self.multiply([m_jj, self.mlp_sbf2(sbf2)])
        m_jj = self.pool_mjj([m, m_jj, angle_idx_2])

        m_ji_2 = self.mlp_ji_2(m, **kwargs)

        m = self.add_mji_2([m_ji_2, m_jj])

        # Aggregation
        m = self.multiply([self.lin_rbf_out(rbf), m])
        h = self.pool_h([h, m, edge_index])

        # Update function f_u
        h = self.res1(h, **kwargs)
        h = self.h_mlp(h, **kwargs)
        h = self.add_res([h, res_h])
        h = self.res2(h, **kwargs)
        h = self.res3(h, **kwargs)

        # Output Module
        y = self.y_mlp(h, **kwargs)
        y = self.y_W(y, **kwargs)

        return h, y