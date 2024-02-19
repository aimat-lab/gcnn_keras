import keras as ks
from keras.layers import Layer, Add, Multiply, Concatenate, Dense
from kgcnn.layers.update import ResidualLayer
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.gather import GatherNodes, GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges as PoolingLocalMessages


class MXMGlobalMP(Layer):

    def __init__(self, units: int = 64, pooling_method="mean", **kwargs):
        """Initialize layer."""
        super(MXMGlobalMP, self).__init__(**kwargs)
        self.dim = units
        self.pooling_method = pooling_method
        self.h_mlp = GraphMLP(self.dim, activation="swish")
        self.res1 = ResidualLayer(self.dim)
        self.res2 = ResidualLayer(self.dim)
        self.res3 = ResidualLayer(self.dim)
        self.mlp = GraphMLP(self.dim, activation="swish")
        self.add_res = Add()

        self.x_edge_mlp = GraphMLP(self.dim, activation="swish")
        self.linear = Dense(self.dim, use_bias=False, activation="linear")

        self.gather = GatherNodes(split_indices=[0, 1], concat_axis=None)
        self.pool = PoolingLocalMessages(pooling_method=pooling_method)
        self.cat = Concatenate(axis=-1)
        self.multiply_edge = Multiply()
        self.add = Add()

    def build(self, input_shape):
        """Build layer."""
        super(MXMGlobalMP, self).build(input_shape)

    def propagate(self, edge_index, x, edge_attr, **kwargs):
        x_i, x_j = self.gather([x, edge_index])

        # Prepare message.
        x_edge = self.cat([x_i, x_j, edge_attr], **kwargs)
        x_edge = self.x_edge_mlp(x_edge, **kwargs)
        edge_attr_lin = self.linear(edge_attr, **kwargs)
        x_edge = self.multiply_edge([edge_attr_lin, x_edge])

        # Pooling here.
        x_p = self.pool([x, x_edge, edge_index])

        # Replace self loops by explicit node update here.
        x_i_p = self.add([x_p, x])

        return x_i_p

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - edges (Tensor): Edge or message embeddings of shape `([M], F)`
                - tensor_index (Tensor): Edge indices referring to nodes of shape `(2, [M])`

        Returns:
            Tensor: Node embeddings.
        """
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

    def get_config(self):
        config = super(MXMGlobalMP, self).get_config()
        config.update({"units": self.dim, "pooling_method": self.pooling_method})
        return config


class MXMLocalMP(Layer):

    def __init__(self, units: int = 64, output_units: int = 1, activation: str = "swish",
                 output_kernel_initializer: str = "zeros", pooling_method: str = "sum", **kwargs):
        super(MXMLocalMP, self).__init__(**kwargs)
        self.dim = units
        self.output_dim = output_units
        self.activation = activation
        self.pooling_method = pooling_method
        self.h_mlp = GraphMLP(self.dim, activation=activation)

        self.mlp_kj = GraphMLP([self.dim], activation=activation)
        self.mlp_ji_1 = GraphMLP([self.dim], activation=activation)
        self.mlp_ji_2 = GraphMLP([self.dim], activation=activation)
        self.mlp_jj = GraphMLP([self.dim], activation=activation)

        self.mlp_sbf1 = GraphMLP([self.dim, self.dim], activation=activation)
        self.mlp_sbf2 = GraphMLP([self.dim, self.dim], activation=activation)
        self.lin_rbf1 = Dense(self.dim, use_bias=False, activation="linear")
        self.lin_rbf2 = Dense(self.dim, use_bias=False, activation="linear")

        self.res1 = ResidualLayer(self.dim)
        self.res2 = ResidualLayer(self.dim)
        self.res3 = ResidualLayer(self.dim)

        self.lin_rbf_out = Dense(self.dim, use_bias=False, activation="linear")

        # Fix for kgcnn==4.0.1: removed overwrite mlp here. Should not change model but prevents unused layers.

        self.y_mlp = GraphMLP([self.dim, self.dim, self.dim], activation=activation)
        self.y_W = Dense(self.output_dim, activation="linear",
                         kernel_initializer=output_kernel_initializer)
        self.add_res = Add()

        self.gather_nodes = GatherNodes(split_indices=[0, 1], concat_axis=None)
        self.cat = Concatenate()
        self.multiply = Multiply()
        self.gather_mkj = GatherNodesOutgoing()
        self.gather_mjj = GatherNodesOutgoing()
        self.pool_mkj = PoolingLocalMessages(pooling_method=pooling_method)
        self.pool_mjj = PoolingLocalMessages(pooling_method=pooling_method)
        self.pool_h = PoolingLocalMessages(pooling_method=pooling_method)
        self.add_mji_1 = Add()
        self.add_mji_2 = Add()
        
    def build(self, input_shape):
        super(MXMLocalMP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [h, rbf, sbf1, sbf2, edge_index, angle_idx_1, angle_idx_2]

                - h (Tensor): Node embeddings of shape `([N], F)`
                - rbf (Tensor): Radial basis functions of shape `([M], F)`
                - sbf1 (Tensor): Spherical basis functions of shape `([K], F)`
                - sbf2 (Tensor): Spherical basis functions of shape `([K], F)`
                - edge_index (Tensor): Edge indices of shape `(2, [M])`
                - angle_idx_1 (Tensor): Angle 1 indices of shape `(2, [K])`
                - angle_idx_2 (Tensor): Angle 2 indices of shape `(2, [K])`

        Returns:
            Tensor: Node embeddings.
        """
        h, rbf, sbf1, sbf2, edge_index, angle_idx_1, angle_idx_2 = inputs
        res_h = h

        # Integrate the Cross Layer Mapping inside the Local Message Passing
        h = self.h_mlp(h, **kwargs)

        # Message Passing 1
        hi, hj = self.gather_nodes([h, edge_index])
        m = self.cat([hi, hj, rbf])

        m_kj = self.mlp_kj(m, **kwargs)
        w_rbf1 = self.lin_rbf1(rbf, **kwargs)
        m_kj = self.multiply([m_kj, w_rbf1])
        m_kj = self.gather_mkj([m_kj, angle_idx_1])
        sw_sbf1 = self.mlp_sbf1(sbf1, **kwargs)
        m_kj = self.multiply([m_kj, sw_sbf1])
        m_kj = self.pool_mkj([m, m_kj, angle_idx_1])

        m_ji_1 = self.mlp_ji_1(m, **kwargs)

        m = self.add_mji_1([m_ji_1, m_kj])

        # Message Passing 2 (index jj denotes j'i in the main paper)
        m_jj = self.mlp_jj(m, **kwargs)
        w_rbf2 = self.lin_rbf2(rbf, **kwargs)
        m_jj = self.multiply([m_jj, w_rbf2])
        m_jj = self.gather_mjj([m_jj, angle_idx_2])
        sw_sbf2 = self.mlp_sbf2(sbf2, **kwargs)
        m_jj = self.multiply([m_jj, sw_sbf2])
        m_jj = self.pool_mjj([m, m_jj, angle_idx_2])

        m_ji_2 = self.mlp_ji_2(m, **kwargs)

        m = self.add_mji_2([m_ji_2, m_jj])

        # Aggregation
        w_rbf = self.lin_rbf_out(rbf, **kwargs)
        m = self.multiply([w_rbf, m])
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

    def get_config(self):
        config = super(MXMLocalMP, self).get_config()
        out_conf = self.y_W.get_config()
        config.update({"units": self.dim, "output_units": self.output_dim,
                       "activation": ks.activations.serialize(ks.activations.get(self.activation)),
                       "output_kernel_initializer": out_conf["kernel_initializer"],
                       "pooling_method": self.pooling_method})
        return config
