import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.disjoint.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.disjoint.pooling import PoolingLocalEdges, PoolingWeightedLocalEdges, PoolingGlobalEdges, \
    PoolingNodes
from kgcnn.ops.activ import kgcnn_custom_act


class GCN(ks.layers.Layer):
    r"""
    Graph convolution according to Kipf et al.
    
    Computes graph conv as $\sigma(A_s*(WX+b))$ where $A_s$ is the precomputed and scaled adjacency matrix.
    The scaled adjacency matrix is defined by $A_s = D^{-0.5} (A + I) D{^-0.5}$ with the degree matrix $D$.
    In place of $A_s$, this layers uses edge features (that are the entries of $A_s$) and edge indices.
    $A_s$ is considered pre-scaled, this is not done by this layer.
    If no scaled edge features are available, you could consider use e.g. "segment_mean", or normalize_by_weights to
    obtain a similar behaviour that is expected by a pre-scaled adjacency matrix input.
    Edge features must be possible to broadcast to node features. Ideally they have shape (...,1).
    
    Args:
        units (int): Output dimension/ units of dense layer.
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
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        is_sorted (bool): If edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
            In this case the edge features are considered weights of dimension (...,1) and are summed for each node.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 node_indexing='batch',
                 pooling_method='segment_sum',
                 is_sorted=False,
                 has_unconnected=True,
                 normalize_by_weights=False,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.normalize_by_weights = normalize_by_weights
        self.partition_type = partition_type
        self.pooling_method = pooling_method
        self.has_unconnected = has_unconnected
        self.is_sorted = is_sorted

        if activation is None and 'leaky_relu' in kgcnn_custom_act:
            activation = {"class_name": "leaky_relu", "config": {"alpha": 0.2}}
        elif activation is None:
            activation = "relu"
        self.units = units
        self.use_bias = use_bias
        self.gcn_activation = tf.keras.activations.get(activation)
        self.gcn_kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.gcn_bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.gcn_kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.gcn_bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.gcn_activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.gcn_kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.gcn_bias_constraint = tf.keras.constraints.get(bias_constraint)

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}

        # Layers
        self.lay_gather = GatherNodesOutgoing(node_indexing=self.node_indexing, partition_type=self.partition_type)
        self.lay_dense = ks.layers.Dense(units=self.units, use_bias=self.use_bias, activation='linear',
                                         **kernel_args)
        self.lay_pool = PoolingWeightedLocalEdges(pooling_method=self.pooling_method, is_sorted=self.is_sorted,
                                                  has_unconnected=self.has_unconnected,
                                                  node_indexing=self.node_indexing,
                                                  normalize_by_weights=self.normalize_by_weights,
                                                  partition_type=self.partition_type)
        self.lay_act = ks.layers.Activation(activation)

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Edge edge_indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.
        
        Returns:
            features (tf.tensor): A list of updated node features.
            Output shape is (batch*None,F).
        """
        node, node_len, edges, edge_len, edge_index = inputs
        no = self.lay_dense(node)
        no = self.lay_gather([no, node_len, edge_index, edge_len])
        nu = self.lay_pool([node, node_len, no, edge_len, edge_index, edges])  # Summing for each node connection
        out = self.lay_act(nu)
        return out

    def get_config(self):
        """Update config."""
        config = super(GCN, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "normalize_by_weights": self.normalize_by_weights,
                       "pooling_method": self.pooling_method,
                       "has_unconnected": self.has_unconnected,
                       "is_sorted": self.is_sorted,
                       "partition_type": self.partition_type,
                       "units": self.units,
                       "use_bias": self.use_bias,
                       "activation": tf.keras.activations.serialize(self.gcn_activation),
                       "kernel_regularizer": tf.keras.regularizers.serialize(self.gcn_kernel_regularizer),
                       "bias_regularizer": tf.keras.regularizers.serialize(self.gcn_bias_regularizer),
                       "activity_regularizer": tf.keras.regularizers.serialize(self.gcn_activity_regularizer),
                       "kernel_constraint": tf.keras.constraints.serialize(self.gcn_kernel_constraint),
                       "bias_constraint": tf.keras.constraints.serialize(self.gcn_bias_constraint),
                       "kernel_initializer": tf.keras.initializers.serialize(self.gcn_kernel_initializer),
                       "bias_initializer": tf.keras.initializers.serialize(self.gcn_bias_initializer)
                       })
        return config


class SchNetCFconv(ks.layers.Layer):
    """
    Continuous filter convolution of SchNet. Assumes disjoint graph representation.
    
    Edges are processed by 2 Dense layers, multiplied on outgoing node features and pooled for ingoing node.
    
    Args:
        units (int): Units for Dense layer.
        use_bias (bool): Use bias. Default is True.
        activation (str): Activation function. Default is 'shifted_softplus' with fall-back 'selu'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        cfconv_pool (str): Pooling method. Default is 'segment_sum'.
        is_sorted (bool): If edge edge_indices are sorted. Default is True.
        has_unconnected (bool): If graph has unconnected nodes. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
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
                 cfconv_pool='segment_sum',
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='batch',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.is_sorted = is_sorted
        self.partition_type = partition_type
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing

        self.units = units
        self.use_bias = use_bias
        if activation is None and 'shifted_softplus' in kgcnn_custom_act:
            activation = 'shifted_softplus'
        elif activation is None:
            activation = "selu"
        self.cfc_activation = tf.keras.activations.get(activation)
        self.cfc_kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.cfc_bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.cfc_kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.cfc_bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.cfc_activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.cfc_kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.cfc_bias_constraint = tf.keras.constraints.get(bias_constraint)

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = ks.layers.Dense(units=self.units, activation=activation, use_bias=self.use_bias,
                                          **kernel_args)
        self.lay_dense2 = ks.layers.Dense(units=self.units, activation='linear', use_bias=self.use_bias,
                                          **kernel_args)
        self.lay_sum = PoolingLocalEdges(pooling_method=self.cfconv_pool,
                                         is_sorted=self.is_sorted,
                                         has_unconnected=self.has_unconnected,
                                         partition_type=self.partition_type,
                                         node_indexing=self.node_indexing)
        self.gather_n = GatherNodesOutgoing(node_indexing=self.node_indexing, partition_type=self.partition_type)

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.
        
        Returns:
            node_update (tf.tensor): Updated node features of shape (batch*None,F)
        """
        node, bn, edge, edge_len, indexlist = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2exp = self.gather_n([node, bn, indexlist, edge_len])
        x = node2exp * x
        x = self.lay_sum([node, bn, x, edge_len, indexlist])
        return x

    def get_config(self):
        """Update layer config."""
        config = super(SchNetCFconv, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "use_bias": self.use_bias,
                       "units": self.units,
                       "activation": tf.keras.activations.serialize(self.cfc_activation),
                       "kernel_regularizer": tf.keras.regularizers.serialize(self.cfc_kernel_regularizer),
                       "bias_regularizer": tf.keras.regularizers.serialize(self.cfc_bias_regularizer),
                       "activity_regularizer": tf.keras.regularizers.serialize(self.cfc_activity_regularizer),
                       "kernel_constraint": tf.keras.constraints.serialize(self.cfc_kernel_constraint),
                       "bias_constraint": tf.keras.constraints.serialize(self.cfc_bias_constraint),
                       "kernel_initializer": tf.keras.initializers.serialize(self.cfc_kernel_initializer),
                       "bias_initializer": tf.keras.initializers.serialize(self.cfc_bias_initializer)
                       })
        return config


class SchNetInteraction(ks.layers.Layer):
    """
    Schnet interaction block, which uses the continuous filter convolution from SchNetCFconv.

    Args:
        units (int): Dimension of node embedding. Default is 64.
        use_bias (bool): Use bias in last layers. Default is True.
        activation (str): Activation function. Default is 'shifted_softplus' with fall-back 'selu'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        cfconv_pool (str): Pooling method information for SchNetCFconv layer. Default is'segment_sum'.
        is_sorted (bool): Whether node indices are sorted. Default is False.
        has_unconnected (bool): Whether graph has unconnected nodes. Default is False.
        partition_type (str): Partition type of the partition information. Default is row_length".
        node_indexing (str): Indexing information. Whether indices refer to per sample or per batch. Default is "batch".
    """

    def __init__(self,
                 units=128,
                 use_bias=True,
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 cfconv_pool='segment_sum',
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='batch',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.partition_type = partition_type
        self.node_indexing = node_indexing
        self.cfconv_pool = cfconv_pool
        self.use_bias = use_bias
        self.units = units
        if activation is None and 'shifted_softplus' in kgcnn_custom_act:
            activation = 'shifted_softplus'
        elif activation is None:
            activation = "selu"
        self.schnet_activation = tf.keras.activations.get(activation)
        self.schnet_kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.schnet_bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.schnet_kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.schnet_bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.schnet_activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.schnet_kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.schnet_bias_constraint = tf.keras.constraints.get(bias_constraint)

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layers
        self.lay_cfconv = SchNetCFconv(units=self.units, activation=activation, use_bias=self.use_bias,
                                       cfconv_pool=self.cfconv_pool, has_unconnected=self.has_unconnected,
                                       is_sorted=self.is_sorted, partition_type=self.partition_type,
                                       node_indexing=self.node_indexing, **kernel_args)
        self.lay_dense1 = ks.layers.Dense(units=self.units, activation='linear', use_bias=False, **kernel_args)
        self.lay_dense2 = ks.layers.Dense(units=self.units, activation=activation, use_bias=self.use_bias,
                                          **kernel_args)
        self.lay_dense3 = ks.layers.Dense(units=self.units, activation='linear', use_bias=self.use_bias,
                                          **kernel_args)
        self.lay_add = ks.layers.Add()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate node update.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.

        Returns:
            node_update (tf.tensor): Updated node features.
        """
        node, bn, edge, edge_len, indexlist = inputs
        x = self.lay_dense1(node)
        x = self.lay_cfconv([x, bn, edge, edge_len, indexlist])
        x = self.lay_dense2(x)
        x = self.lay_dense3(x)
        out = self.lay_add([node, x])
        return out

    def get_config(self):
        config = super(SchNetInteraction, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool,
                       "is_sorted}": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "node_indexing": self.node_indexing,
                       "units": self.units,
                       "use_bias": self.use_bias,
                       "activation": tf.keras.activations.serialize(self.schnet_activation),
                       "kernel_regularizer": tf.keras.regularizers.serialize(self.schnet_kernel_regularizer),
                       "bias_regularizer": tf.keras.regularizers.serialize(self.schnet_bias_regularizer),
                       "activity_regularizer": tf.keras.regularizers.serialize(self.schnet_activity_regularizer),
                       "kernel_constraint": tf.keras.constraints.serialize(self.schnet_kernel_constraint),
                       "bias_constraint": tf.keras.constraints.serialize(self.schnet_bias_constraint),
                       "kernel_initializer": tf.keras.initializers.serialize(self.schnet_kernel_initializer),
                       "bias_initializer": tf.keras.initializers.serialize(self.schnet_bias_initializer)
                       })
        return config


class MEGnetBlock(ks.layers.Layer):
    """
    Megnet Block.

    Args:
        node_embed (list, optional): List of node embedding dimension. Defaults to [16,16,16].
        edge_embed (list, optional): List of edge embedding dimension. Defaults to [16,16,16].
        env_embed (list, optional): List of environment embedding dimension. Defaults to [16,16,16].
        use_bias (bool, optional): Use bias. Defaults to True.
        activation (str): Activation function. Default is 'softplus2' with fall-back 'selu'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        is_sorted (bool, optional): Edge index list is sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        partition_type (str): Partition type of the partition information. Default is row_length".
        node_indexing (str): Indexing information. Whether indices refer to per sample or per batch. Default is "batch".
        **kwargs
    """

    def __init__(self, node_embed=None,
                 edge_embed=None,
                 env_embed=None,
                 use_bias=True,
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='batch',
                 **kwargs):
        """Initialize layer."""
        super(MEGnetBlock, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.partition_type = partition_type
        self.node_indexing = node_indexing

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
        self.megnet_activation = tf.keras.activations.get(activation)
        self.megnet_kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.megnet_bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.megnet_kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.megnet_bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.megnet_activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.megnet_kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.megnet_bias_constraint = tf.keras.constraints.get(bias_constraint)

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}

        # Node
        self.lay_phi_n = ks.layers.Dense(units=self.node_embed[0], activation=activation, use_bias=self.use_bias,
                                         **kernel_args)
        self.lay_phi_n_1 = ks.layers.Dense(units=self.node_embed[1], activation=activation, use_bias=self.use_bias,
                                           **kernel_args)
        self.lay_phi_n_2 = ks.layers.Dense(units=self.node_embed[2], activation='linear', use_bias=self.use_bias,
                                           **kernel_args)
        self.lay_esum = PoolingLocalEdges(is_sorted=self.is_sorted, has_unconnected=self.has_unconnected,
                                          partition_type=self.partition_type, node_indexing=self.node_indexing)
        self.lay_gather_un = GatherState(partition_type=self.partition_type)
        self.lay_conc_nu = ks.layers.Concatenate(axis=-1)
        # Edge
        self.lay_phi_e = ks.layers.Dense(units=self.edge_embed[0], activation=activation, use_bias=self.use_bias,
                                         **kernel_args)
        self.lay_phi_e_1 = ks.layers.Dense(units=self.edge_embed[1], activation=activation, use_bias=self.use_bias,
                                           **kernel_args)
        self.lay_phi_e_2 = ks.layers.Dense(units=self.edge_embed[2], activation='linear', use_bias=self.use_bias,
                                           **kernel_args)
        self.lay_gather_n = GatherNodes(node_indexing=self.node_indexing, partition_type=self.partition_type)
        self.lay_gather_ue = GatherState(partition_type=self.partition_type)
        self.lay_conc_enu = ks.layers.Concatenate(axis=-1)
        # Environment
        self.lay_usum_e = PoolingGlobalEdges(partition_type=self.partition_type)
        self.lay_usum_n = PoolingNodes(partition_type=self.partition_type)
        self.lay_conc_u = ks.layers.Concatenate(axis=-1)
        self.lay_phi_u = ks.layers.Dense(units=self.env_embed[0], activation=activation, use_bias=self.use_bias,
                                         **kernel_args)
        self.lay_phi_u_1 = ks.layers.Dense(units=self.env_embed[1], activation=activation, use_bias=self.use_bias,
                                           **kernel_args)
        self.lay_phi_u_2 = ks.layers.Dense(units=self.env_embed[2], activation='linear', use_bias=self.use_bias,
                                           **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(MEGnetBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, edges, edge_index, env_input, node_partition, edge_partition]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.
            - graph_state (tf.tensor): Graph state input of shape (batch, F).
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)

        Returns:
            list: vp,ep,up
        """
        # Calculate edge Update
        node_input, edge_input, edge_index_input, env_input, len_node, len_edge = inputs
        e_n = self.lay_gather_n([node_input, len_node, edge_index_input, len_edge])
        e_u = self.lay_gather_ue([env_input, len_edge])
        ec = self.lay_conc_enu([e_n, edge_input, e_u])
        ep = self.lay_phi_e(ec)  # Learning of Update Functions
        ep = self.lay_phi_e_1(ep)  # Learning of Update Functions
        ep = self.lay_phi_e_2(ep)  # Learning of Update Functions
        # Calculate Node update
        vb = self.lay_esum([node_input, len_node, ep, len_edge, edge_index_input])  # Summing for each node connections
        v_u = self.lay_gather_un([env_input, len_node])
        vc = self.lay_conc_nu([vb, node_input, v_u])  # Concatenate node features with new edge updates
        vp = self.lay_phi_n(vc)  # Learning of Update Functions
        vp = self.lay_phi_n_1(vp)  # Learning of Update Functions
        vp = self.lay_phi_n_2(vp)  # Learning of Update Functions
        # Calculate environment update
        es = self.lay_usum_e([ep, len_edge])
        vs = self.lay_usum_n([vp, len_node])
        ub = self.lay_conc_u([es, vs, env_input])
        up = self.lay_phi_u(ub)
        up = self.lay_phi_u_1(up)
        up = self.lay_phi_u_2(up)  # Learning of Update Functions
        return vp, ep, up

    def get_config(self):
        config = super(MEGnetBlock, self).get_config()
        config.update({"is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "node_indexing": self.node_indexing,
                       "node_embed": self.node_embed,
                       "edge_embed": self.edge_embed,
                       "env_embed": self.env_embed,
                       "use_bias": self.use_bias,
                       "activation": tf.keras.activations.serialize(self.megnet_activation),
                       "kernel_regularizer": tf.keras.regularizers.serialize(self.megnet_kernel_regularizer),
                       "bias_regularizer": tf.keras.regularizers.serialize(self.megnet_bias_regularizer),
                       "activity_regularizer": tf.keras.regularizers.serialize(self.megnet_activity_regularizer),
                       "kernel_constraint": tf.keras.constraints.serialize(self.megnet_kernel_constraint),
                       "bias_constraint": tf.keras.constraints.serialize(self.megnet_bias_constraint),
                       "kernel_initializer": tf.keras.initializers.serialize(self.megnet_kernel_initializer),
                       "bias_initializer": tf.keras.initializers.serialize(self.megnet_bias_initializer)
                       })
        return config
