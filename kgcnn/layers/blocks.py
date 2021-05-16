import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.keras import Dense, Activation, Add, Multiply, Concatenate
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingWeightedLocalEdges, PoolingGlobalEdges, \
    PoolingNodes
from kgcnn.ops.activ import kgcnn_custom_act


class MEGnetBlock(GraphBaseLayer):
    """Megnet Block.

    Args:
        node_embed (list, optional): List of node embedding dimension. Defaults to [16,16,16].
        edge_embed (list, optional): List of edge embedding dimension. Defaults to [16,16,16].
        env_embed (list, optional): List of environment embedding dimension. Defaults to [16,16,16].
        pooling_method (str): Pooling method information for layer. Default is 'mean'.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str): Activation function. Default is 'softplus2' with fall-back 'selu'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, node_embed=None,
                 edge_embed=None,
                 env_embed=None,
                 pooling_method="mean",
                 use_bias=True,
                 activation=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 **kwargs):
        """Initialize layer."""
        super(MEGnetBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method

        if node_embed is None:
            node_embed = [16, 16, 16]
        if env_embed is None:
            env_embed = [16, 16, 16]
        if edge_embed is None:
            edge_embed = [16, 16, 16]
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.env_embed = env_embed
        self.use_bias = use_bias
        if activation is None and 'softplus2' in kgcnn_custom_act:
            activation = 'softplus2'
        elif activation is None:
            activation = "selu"

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}
        mlp_args = {"input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate}
        mlp_args.update(kernel_args)
        pool_args = {"pooling_method": self.pooling_method}
        pool_args.update(self._all_kgcnn_info)
        gather_args = self._all_kgcnn_info

        # Node
        self.lay_phi_n = Dense(units=self.node_embed[0], activation=activation, **mlp_args)
        self.lay_phi_n_1 = Dense(units=self.node_embed[1], activation=activation, **mlp_args)
        self.lay_phi_n_2 = Dense(units=self.node_embed[2], activation='linear', **mlp_args)
        self.lay_esum = PoolingLocalEdges(**pool_args)
        self.lay_gather_un = GatherState(**gather_args)
        self.lay_conc_nu = Concatenate(axis=-1, input_tensor_type=self.input_tensor_type)
        # Edge
        self.lay_phi_e = Dense(units=self.edge_embed[0], activation=activation, **mlp_args)
        self.lay_phi_e_1 = Dense(units=self.edge_embed[1], activation=activation, **mlp_args)
        self.lay_phi_e_2 = Dense(units=self.edge_embed[2], activation='linear', **mlp_args)
        self.lay_gather_n = GatherNodes(**gather_args)
        self.lay_gather_ue = GatherState(**gather_args)
        self.lay_conc_enu = Concatenate(axis=-1, input_tensor_type=self.input_tensor_type)
        # Environment
        self.lay_usum_e = PoolingGlobalEdges(**pool_args)
        self.lay_usum_n = PoolingNodes(**pool_args)
        self.lay_conc_u = Concatenate(axis=-1, input_tensor_type="tensor")
        self.lay_phi_u = ks.layers.Dense(units=self.env_embed[0], activation=activation, **kernel_args)
        self.lay_phi_u_1 = ks.layers.Dense(units=self.env_embed[1], activation=activation, **kernel_args)
        self.lay_phi_u_2 = ks.layers.Dense(units=self.env_embed[2], activation='linear', **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(MEGnetBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index, state]

            - nodes (tf.ragged): Node embeddings of shape (batch, [N], F)
            - edges (tf.ragged): Edge or message embeddings of shape (batch, [M], F)
            - edge_index (tf.ragged): Edge indices of shape (batch, [M], 2)
            - state (tf.tensor): State information for the graph, a single tensor of shape (batch, F)

        Returns:
            node_update: Updated node embeddings of shape (batch, [N], F)
        """
        # Calculate edge Update
        node_input, edge_input, edge_index_input, env_input = inputs
        e_n = self.lay_gather_n([node_input, edge_index_input])
        e_u = self.lay_gather_ue([env_input, edge_input])
        ec = self.lay_conc_enu([e_n, edge_input, e_u])
        ep = self.lay_phi_e(ec)  # Learning of Update Functions
        ep = self.lay_phi_e_1(ep)  # Learning of Update Functions
        ep = self.lay_phi_e_2(ep)  # Learning of Update Functions
        # Calculate Node update
        vb = self.lay_esum([node_input, ep, edge_index_input])  # Summing for each node connections
        v_u = self.lay_gather_un([env_input, node_input])
        vc = self.lay_conc_nu([vb, node_input, v_u])  # Concatenate node features with new edge updates
        vp = self.lay_phi_n(vc)  # Learning of Update Functions
        vp = self.lay_phi_n_1(vp)  # Learning of Update Functions
        vp = self.lay_phi_n_2(vp)  # Learning of Update Functions
        # Calculate environment update
        es = self.lay_usum_e(ep)
        vs = self.lay_usum_n(vp)
        ub = self.lay_conc_u([es, vs, env_input])
        up = self.lay_phi_u(ub)
        up = self.lay_phi_u_1(up)
        up = self.lay_phi_u_2(up)  # Learning of Update Functions
        return vp, ep, up

    def get_config(self):
        config = super(MEGnetBlock, self).get_config()
        config.update({"pooling_method": self.pooling_method, "node_embed": self.node_embed, "use_bias": self.use_bias,
                       "edge_embed": self.edge_embed, "env_embed": self.env_embed})
        config_dense = self.lay_phi_n.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: config_dense[x]})
        return config


class DimNetOutputBlock(GraphBaseLayer):
    """DimNetOutputBlock.

    Args:
        emb_size (list): List of node embedding dimension.
        out_emb_size (list): List of edge embedding dimension.
        num_dense (list): List of environment embedding dimension.
        num_targets (int): Number of output target dimension. Defaults to 12.
        use_bias (bool, optional): Use bias. Defaults to True.
        kernel_initializer: Initializer for kernels. Default is 'orthogonal'.
        output_kernel_initializer: Initializer for last kernel. Default is 'zeros'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        activation (str): Activation function. Default is 'softplus2' with fall-back 'selu'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        pooling_method (str): Pooling method information for layer. Default is 'mean'.
    """

    def __init__(self, emb_size,
                 out_emb_size,
                 num_dense,
                 num_targets=12,
                 use_bias=True,
                 output_kernel_initializer="zeros", kernel_initializer='orthogonal', bias_initializer='zeros',
                 activation=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 pooling_method="sum",
                 **kwargs):
        """Initialize layer."""
        super(DimNetOutputBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size
        self.num_dense = num_dense
        self.num_targets = num_targets
        self.use_bias = use_bias

        if activation is None and 'swish' in kgcnn_custom_act:
            activation = 'swish'
        elif activation is None:
            activation = "selu"

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_initializer": bias_initializer,
                       "bias_regularizer": bias_regularizer, "bias_constraint": bias_constraint, }
        mlp_args = {"input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate}
        mlp_args.update(kernel_args)
        pool_args = {"pooling_method": self.pooling_method}
        pool_args.update(self._all_kgcnn_info)

        self.dense_rbf = Dense(emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.up_projection = Dense(out_emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.dense_mlp = MLP([out_emb_size] * num_dense, activation=activation, kernel_initializer=kernel_initializer,
                             use_bias=use_bias, **mlp_args)
        self.dimnet_mult = Multiply(input_tensor_type=self.input_tensor_type)
        self.pool = PoolingLocalEdges(**pool_args)
        self.dense_final = Dense(num_targets, use_bias=False, kernel_initializer=output_kernel_initializer,
                                 **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetOutputBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index, state]

            - nodes (tf.ragged): Node embeddings of shape (batch, [N], F)
            - edges (tf.ragged): Edge or message embeddings of shape (batch, [M], F)
            - rbf (tf.ragged): Edge distance basis of shape (batch, [M], F)
            - edge_index (tf.ragged): Node indices of shape (batch, [M], 2)

        Returns:
            node_update (tf.ragged): Updated node embeddings.
        """
        # Calculate edge Update
        n_atoms, x, rbf, idnb_i = inputs
        g = self.dense_rbf(rbf)
        x = self.dimnet_mult([g, x])
        x = self.pool([n_atoms, x, idnb_i])
        x = self.up_projection(x)
        x = self.dense_mlp(x)
        x = self.dense_final(x)
        return x

    def get_config(self):
        config = super(DimNetOutputBlock, self).get_config()
        conf_mlp = self.dense_mlp.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_mlp[x][0]})
        conf_dense_output = self.dense_final.get_config()
        config.update({"output_kernel_initializer": conf_dense_output["kernel_initializer"]})
        config.update({"pooling_method": self.pooling_method, "use_bias": self.use_bias})
        config.update({"emb_size": self.emb_size, "out_emb_size": self.out_emb_size, "num_dense": self.num_dense,
                       "num_targets": self.num_targets})
        return config
