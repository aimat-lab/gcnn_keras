import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate
from kgcnn.layers.gather import GatherNodes, GatherState
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingGlobalEdges, PoolingNodes
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='MEGnetBlock')
class MEGnetBlock(GraphBaseLayer):
    r"""Convolutional unit of `MegNet <https://github.com/materialsvirtuallab/megnet>`_ called MegNet Block.

    Args:
        node_embed (list, optional): List of node embedding dimension. Defaults to [16,16,16].
        edge_embed (list, optional): List of edge embedding dimension. Defaults to [16,16,16].
        env_embed (list, optional): List of environment embedding dimension. Defaults to [16,16,16].
        pooling_method (str): Pooling method information for layer. Default is 'mean'.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str): Activation function. Default is 'kgcnn>softplus2'.
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
                 activation='kgcnn>softplus2',
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
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}

        # Node
        self.lay_phi_n = DenseEmbedding(units=self.node_embed[0], activation=activation, **kernel_args)
        self.lay_phi_n_1 = DenseEmbedding(units=self.node_embed[1], activation=activation, **kernel_args)
        self.lay_phi_n_2 = DenseEmbedding(units=self.node_embed[2], activation='linear', **kernel_args)
        self.lay_esum = PoolingLocalEdges(pooling_method=self.pooling_method)
        self.lay_gather_un = GatherState()
        self.lay_conc_nu = LazyConcatenate(axis=-1)
        # Edge
        self.lay_phi_e = DenseEmbedding(units=self.edge_embed[0], activation=activation, **kernel_args)
        self.lay_phi_e_1 = DenseEmbedding(units=self.edge_embed[1], activation=activation, **kernel_args)
        self.lay_phi_e_2 = DenseEmbedding(units=self.edge_embed[2], activation='linear', **kernel_args)
        self.lay_gather_n = GatherNodes()
        self.lay_gather_ue = GatherState()
        self.lay_conc_enu = LazyConcatenate(axis=-1)
        # Environment
        self.lay_usum_e = PoolingGlobalEdges(pooling_method=self.pooling_method)
        self.lay_usum_n = PoolingNodes(pooling_method=self.pooling_method)
        self.lay_conc_u = LazyConcatenate(axis=-1)
        self.lay_phi_u = ks.layers.Dense(units=self.env_embed[0], activation=activation, **kernel_args)
        self.lay_phi_u_1 = ks.layers.Dense(units=self.env_embed[1], activation=activation, **kernel_args)
        self.lay_phi_u_2 = ks.layers.Dense(units=self.env_embed[2], activation='linear', **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(MEGnetBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index, state]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
                - state (tf.Tensor): State information for the graph, a single tensor of shape (batch, F)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F)
        """
        # Calculate edge Update
        node_input, edge_input, edge_index_input, env_input = inputs
        e_n = self.lay_gather_n([node_input, edge_index_input], **kwargs)
        e_u = self.lay_gather_ue([env_input, edge_input], **kwargs)
        ec = self.lay_conc_enu([e_n, edge_input, e_u], **kwargs)
        ep = self.lay_phi_e(ec, **kwargs)  # Learning of Update Functions
        ep = self.lay_phi_e_1(ep, **kwargs)  # Learning of Update Functions
        ep = self.lay_phi_e_2(ep, **kwargs)  # Learning of Update Functions
        # Calculate Node update
        vb = self.lay_esum([node_input, ep, edge_index_input], **kwargs)  # Summing for each node connections
        v_u = self.lay_gather_un([env_input, node_input], **kwargs)
        vc = self.lay_conc_nu([vb, node_input, v_u], **kwargs)  # LazyConcatenate node features with new edge updates
        vp = self.lay_phi_n(vc, **kwargs)  # Learning of Update Functions
        vp = self.lay_phi_n_1(vp, **kwargs)  # Learning of Update Functions
        vp = self.lay_phi_n_2(vp, **kwargs)  # Learning of Update Functions
        # Calculate environment update
        es = self.lay_usum_e(ep, **kwargs)
        vs = self.lay_usum_n(vp, **kwargs)
        ub = self.lay_conc_u([es, vs, env_input], **kwargs)
        up = self.lay_phi_u(ub, **kwargs)
        up = self.lay_phi_u_1(up, **kwargs)
        up = self.lay_phi_u_2(up, **kwargs)  # Learning of Update Functions
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
