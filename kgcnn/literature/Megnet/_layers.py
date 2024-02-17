from keras.layers import Layer, Dense, Concatenate
from kgcnn.layers.gather import GatherNodes, GatherState
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.ops.activ import softplus2


PoolingGlobalEdges = PoolingNodes


class MEGnetBlock(Layer):
    r"""Convolutional unit of `MegNet <https://github.com/materialsvirtuallab/megnet>`_ called MegNet Block."""

    def __init__(self,
                 node_embed=None,
                 edge_embed=None,
                 env_embed=None,
                 pooling_method="mean",
                 use_bias=True,
                 activation='kgcnn>softplus2',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 **kwargs):
        """Initialize layer.

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
        super(MEGnetBlock, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default.
        if activation in ["kgcnn>softplus2"]:
            activation = {"class_name": "function", "config": "kgcnn>softplus2"}
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
        self.lay_phi_n = Dense(units=self.node_embed[0], activation=activation, **kernel_args)
        self.lay_phi_n_1 = Dense(units=self.node_embed[1], activation=activation, **kernel_args)
        self.lay_phi_n_2 = Dense(units=self.node_embed[2], activation='linear', **kernel_args)
        self.lay_esum = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.lay_gather_un = GatherState()
        self.lay_conc_nu = Concatenate(axis=-1)
        # Edge
        self.lay_phi_e = Dense(units=self.edge_embed[0], activation=activation, **kernel_args)
        self.lay_phi_e_1 = Dense(units=self.edge_embed[1], activation=activation, **kernel_args)
        self.lay_phi_e_2 = Dense(units=self.edge_embed[2], activation='linear', **kernel_args)
        self.lay_gather_n = GatherNodes()
        self.lay_gather_ue = GatherState()
        self.lay_conc_enu = Concatenate(axis=-1)
        # Environment
        self.lay_usum_e = PoolingGlobalEdges(pooling_method=self.pooling_method)
        self.lay_usum_n = PoolingNodes(pooling_method=self.pooling_method)
        self.lay_conc_u = Concatenate(axis=-1)
        self.lay_phi_u = Dense(units=self.env_embed[0], activation=activation, **kernel_args)
        self.lay_phi_u_1 = Dense(units=self.env_embed[1], activation=activation, **kernel_args)
        self.lay_phi_u_2 = Dense(units=self.env_embed[2], activation='linear', **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(MEGnetBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, tensor_index, state, batch_id_node, batch_id_edge, count_nodes, count_edges]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - tensor_index (Tensor): Edge indices referring to nodes of shape (2, [M])
                - state (Tensor): State information for the graph, a single tensor of shape (batch, F)
                - graph_id_node (Tensor): ID tensor of batch assignment in disjoint graph of shape `([N], )` .
                - graph_id_edge (Tensor): ID tensor of batch assignment in disjoint graph of shape `([M], )` .
                - nodes_count (Tensor): Tensor of number of nodes for each graph of shape `(batch, )` .
                - edges_count (Tensor): Tensor of number of edges for each graph of shape `(batch, )` .

        Returns:
            Tensor: Updated node embeddings of shape ([N], F)
        """
        # Calculate edge Update
        node_input, edge_input, edge_index_input, env_input, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs
        e_n = self.lay_gather_n([node_input, edge_index_input], **kwargs)
        e_u = self.lay_gather_ue([env_input, batch_id_edge], **kwargs)
        ec = self.lay_conc_enu([e_n, edge_input, e_u], **kwargs)
        ep = self.lay_phi_e(ec, **kwargs)  # Learning of Update Functions
        ep = self.lay_phi_e_1(ep, **kwargs)  # Learning of Update Functions
        ep = self.lay_phi_e_2(ep, **kwargs)  # Learning of Update Functions
        # Calculate Node update
        vb = self.lay_esum([node_input, ep, edge_index_input], **kwargs)  # Summing for each node connections
        v_u = self.lay_gather_un([env_input, batch_id_node], **kwargs)
        vc = self.lay_conc_nu([vb, node_input, v_u], **kwargs)  # LazyConcatenate node features with new edge updates
        vp = self.lay_phi_n(vc, **kwargs)  # Learning of Update Functions
        vp = self.lay_phi_n_1(vp, **kwargs)  # Learning of Update Functions
        vp = self.lay_phi_n_2(vp, **kwargs)  # Learning of Update Functions
        # Calculate environment update
        es = self.lay_usum_e([count_edges, ep, batch_id_edge], **kwargs)
        vs = self.lay_usum_n([count_nodes, vp, batch_id_node], **kwargs)
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
            if x in config_dense:
                config.update({x: config_dense[x]})
        return config
