import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.keras import Dense, Activation, Add, Multiply, Concatenate
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingWeightedLocalEdges, PoolingGlobalEdges, \
    PoolingNodes
from kgcnn.ops.activ import kgcnn_custom_act


class GCN(GraphBaseLayer):
    r"""Graph convolution according to Kipf et al.
    
    Computes graph conv as $\sigma(A_s*(WX+b))$ where $A_s$ is the precomputed and scaled adjacency matrix.
    The scaled adjacency matrix is defined by $A_s = D^{-0.5} (A + I) D{^-0.5}$ with the degree matrix $D$.
    In place of $A_s$, this layers uses edge features (that are the entries of $A_s$) and edge indices.
    $A_s$ is considered pre-scaled, this is not done by this layer.
    If no scaled edge features are available, you could consider use e.g. "segment_mean", or normalize_by_weights to
    obtain a similar behaviour that is expected by a pre-scaled adjacency matrix input.
    Edge features must be possible to broadcast to node features. Ideally they have shape (...,1).
    
    Args:
        units (int): Output dimension/ units of dense layer.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
            In this case the edge features are considered weights of dimension (...,1) and are summed for each node.
        activation (str): Activation. Default is {"class_name": "leaky_relu", "config": {"alpha": 0.2}},
            with fall-back "relu".
        use_bias (bool): Use bias. Default is True.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units,
                 pooling_method='sum',
                 normalize_by_weights=False,
                 activation=None,
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.normalize_by_weights = normalize_by_weights
        self.pooling_method = pooling_method
        self.units = units
        if activation is None and 'leaky_relu' in kgcnn_custom_act:
            activation = {"class_name": "leaky_relu", "config": {"alpha": 0.2}}
        elif activation is None:
            activation = "relu"

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}
        gather_args = self._all_kgcnn_info
        pool_args = {"pooling_method": pooling_method, "normalize_by_weights": normalize_by_weights}
        pool_args.update(self._all_kgcnn_info)

        # Layers
        self.lay_gather = GatherNodesOutgoing(**gather_args)
        self.lay_dense = Dense(units=self.units, activation='linear',
                               input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                               **kernel_args)
        self.lay_pool = PoolingWeightedLocalEdges(**pool_args)
        self.lay_act = Activation(activation, ragged_validate=self.ragged_validate,
                                  input_tensor_type=self.input_tensor_type)

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index]

            - nodes (tf.ragged): Node embeddings of shape (batch, [N], F)
            - edges (tf.ragged): Edge or message embeddings of shape (batch, [M], F)
            - edge_index (tf.ragged): Edge indices of shape (batch, [M], 2)

        Returns:
            embeddings: Node embeddings of shape (batch, [N], F)
        """
        node, edges, edge_index = inputs
        no = self.lay_dense(node)
        no = self.lay_gather([no, edge_index])
        nu = self.lay_pool([node, no, edge_index, edges])  # Summing for each node connection
        out = self.lay_act(nu)
        return out

    def get_config(self):
        """Update config."""
        config = super(GCN, self).get_config()
        config.update({"normalize_by_weights": self.normalize_by_weights,
                       "pooling_method": self.pooling_method, "units": self.units})
        conf_dense = self.lay_dense.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "use_bias"]:
            config.update({x: conf_dense[x]})
        conf_act = self.lay_act.get_config()
        config.update({"activation": conf_act["activation"]})
        return config


class SchNetCFconv(GraphBaseLayer):
    """Continuous filter convolution of SchNet.
    
    Edges are processed by 2 Dense layers, multiplied on outgoing node features and pooled for ingoing node.
    
    Args:
        units (int): Units for Dense layer.
        cfconv_pool (str): Pooling method. Default is 'segment_sum'.
        use_bias (bool): Use bias. Default is True.
        activation (str): Activation function. Default is 'shifted_softplus' with fall-back 'selu'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, units,
                 cfconv_pool='segment_sum',
                 use_bias=True,
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.units = units
        self.use_bias = use_bias

        if activation is None and 'shifted_softplus' in kgcnn_custom_act:
            activation = 'shifted_softplus'
        elif activation is None:
            activation = "selu"

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        pooling_args = {"pooling_method": cfconv_pool}
        pooling_args.update(self._all_kgcnn_info)
        # Layer
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias,
                                input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                **kernel_args)
        self.lay_dense2 = Dense(units=self.units, activation='linear', use_bias=self.use_bias,
                                input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                **kernel_args)
        self.lay_sum = PoolingLocalEdges(**pooling_args)
        self.gather_n = GatherNodesOutgoing(**self._all_kgcnn_info)
        self.lay_mult = Multiply(input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate)

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs: [nodes, edges, edge_index]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edges: Edge or message embeddings of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)
        
        Returns:
            node_update: Updated node features.
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2exp = self.gather_n([node, indexlist])
        x = self.lay_mult([node2exp, x])
        x = self.lay_sum([node, x, indexlist])
        return x

    def get_config(self):
        """Update layer config."""
        config = super(SchNetCFconv, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


class SchNetInteraction(GraphBaseLayer):
    """
    Schnet interaction block, which uses the continuous filter convolution from SchNetCFconv.

    Args:
        units (int): Dimension of node embedding. Default is 128.
        cfconv_pool (str): Pooling method information for SchNetCFconv layer. Default is'segment_sum'.
        use_bias (bool): Use bias in last layers. Default is True.
        activation (str): Activation function. Default is 'shifted_softplus' with fall-back 'selu'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units=128,
                 cfconv_pool='sum',
                 use_bias=True,
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)

        self.cfconv_pool = cfconv_pool
        self.use_bias = use_bias
        self.units = units
        if activation is None and 'shifted_softplus' in kgcnn_custom_act:
            activation = 'shifted_softplus'
        elif activation is None:
            activation = "selu"

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        conv_args = {"units": self.units, "use_bias": use_bias, "activation": activation, "cfconv_pool": cfconv_pool}
        conv_args.update(kernel_args)
        conv_args.update(self._all_kgcnn_info)
        # Layers
        self.lay_cfconv = SchNetCFconv(**conv_args)
        self.lay_dense1 = Dense(units=self.units, activation='linear', use_bias=False,
                                input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                **kernel_args)
        self.lay_dense2 = Dense(units=self.units, activation=activation, use_bias=self.use_bias,
                                input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                **kernel_args)
        self.lay_dense3 = Dense(units=self.units, activation='linear', use_bias=self.use_bias,
                                input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                **kernel_args)
        self.lay_add = Add(input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate)

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate node update.

        Args:
            inputs: [nodes, edges, edge_index]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edges: Edge or message embeddings of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)

        Returns:
            node_update: Updated node embeddings.
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(node)
        x = self.lay_cfconv([x, edge, indexlist])
        x = self.lay_dense2(x)
        x = self.lay_dense3(x)
        out = self.lay_add([node, x])
        return out

    def get_config(self):
        config = super(SchNetInteraction, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units, "use_bias": self.use_bias})
        conf_dense = self.lay_dense2.get_config()
        for x in ["activation", "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                  "kernel_constraint", "bias_constraint", "kernel_initializer", "bias_initializer"]:
            config.update({x: conf_dense[x]})
        return config


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


class ResidualLayer(GraphBaseLayer):
    """Residual Layer as defined by DimNet.

    Args:
        units: Dimension of the kernel.
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str): Activation function. Default is "swish".
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        **kwargs:
    """

    def __init__(self, units,
                 use_bias=True,
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize layer."""
        super(ResidualLayer, self).__init__(**kwargs)
        if activation is None and 'swish' in kgcnn_custom_act:
            activation = 'swish'
        elif activation is None:
            activation = "selu"
        dense_args = {"units": units, "activation": activation, "use_bias": use_bias,
                      "kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                      "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                      "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                      "bias_initializer": bias_initializer}

        self.dense_1 = Dense(**dense_args)
        self.dense_2 = Dense(**dense_args)
        self.add_end = Add()

    def build(self, input_shape):
        """Build layer."""
        super(ResidualLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass

        Args:
            inputs (tf.ragged): Node or edge embedding of shape (batch, [N], F)

        Returns:
            embeddings: Node or edge embedding of shape (batch, [N], F)
        """
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.add_end([inputs, x])
        return x

    def get_config(self):
        config = super(ResidualLayer, self).get_config()
        conf_dense = self.dense_1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias", "units"]:
            config.update({x: conf_dense[x]})
        return config


class DimNetInteractionPPBlock(GraphBaseLayer):
    """DimNetInteractionPPBlock as defined by DimNet.

    Args:
        emb_size: Embedding size used for the messages
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size: Embedding size used inside the basis transformation
        num_before_skip: Number of residual layers in interaction block before skip connection
        num_after_skip: Number of residual layers in interaction block before skip connection
        use_bias (bool, optional): Use bias. Defaults to True.
        pooling_method (str): Pooling method information for layer. Default is 'sum'.
        activation (str): Activation function. Default is "swish".
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'orthogonal'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        **kwargs:
    """

    def __init__(self, emb_size,
                 int_emb_size,
                 basis_emb_size,
                 num_before_skip,
                 num_after_skip,
                 use_bias=True,
                 pooling_method="sum",
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='orthogonal',
                 bias_initializer='zeros',
                 **kwargs):
        super(DimNetInteractionPPBlock, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.pooling_method = pooling_method
        self.emb_size = emb_size
        self.int_emb_size = int_emb_size
        self.basis_emb_size = basis_emb_size
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        if activation is None and 'swish' in kgcnn_custom_act:
            activation = 'swish'
        elif activation is None:
            activation = "selu"

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        pool_args = {"pooling_method": pooling_method}
        pool_args.update(self._all_kgcnn_info)
        gather_args = self._all_kgcnn_info

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_rbf2 = Dense(emb_size, use_bias=False, **kernel_args)
        self.dense_sbf1 = Dense(basis_emb_size, use_bias=False, **kernel_args)
        self.dense_sbf2 = Dense(int_emb_size, use_bias=False, **kernel_args)

        # Dense transformations of input messages
        self.dense_ji = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)
        self.dense_kj = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Embedding projections for interaction triplets
        self.down_projection = Dense(int_emb_size, activation=activation, use_bias=False, **kernel_args)
        self.up_projection = Dense(emb_size, activation=activation, use_bias=False, **kernel_args)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))
        self.final_before_skip = Dense(emb_size, activation=activation, use_bias=True, **kernel_args)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True, **kernel_args))

        self.lay_add1 = Add()
        self.lay_add2 = Add()
        self.lay_mult1 = Multiply()
        self.lay_mult2 = Multiply()

        self.lay_gather = GatherNodesOutgoing(**gather_args)  # Are edges here
        self.lay_pool = PoolingLocalEdges(**pool_args)

    def call(self, inputs, **kwargs):
        x, rbf, sbf, id_expand = inputs

        # Initial transformation
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)

        # Transform via Bessel basis
        rbf = self.dense_rbf1(rbf)
        rbf = self.dense_rbf2(rbf)
        x_kj = self.lay_mult1([x_kj, rbf])

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.down_projection(x_kj)
        x_kj = self.lay_gather([x_kj, id_expand])

        # Transform via 2D spherical basis
        sbf = self.dense_sbf1(sbf)
        sbf = self.dense_sbf2(sbf)
        x_kj = self.lay_mult1([x_kj, sbf])

        # Aggregate interactions and up-project embeddings
        x_kj = self.lay_pool([rbf, x_kj, id_expand])
        x_kj = self.up_projection(x_kj)

        # Transformations before skip connection
        x2 = self.lay_add1([x_ji, x_kj])
        for layer in self.layers_before_skip:
            x2 = layer(x2)
        x2 = self.final_before_skip(x2)

        # Skip connection
        x = self.lay_add2([x, x2])

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x)

        return x

    def get_config(self):
        config = super(DimNetInteractionPPBlock, self).get_config()
        config.update({"use_bias": self.use_bias, "pooling_method": self.pooling_method, "emb_size": self.emb_size,
                       "int_emb_size": self.int_emb_size, "basis_emb_size": self.basis_emb_size,
                       "num_before_skip": self.num_before_skip, "num_after_skip": self.num_after_skip})
        conf_dense = self.dense_ji.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_dense[x]})
        return config
