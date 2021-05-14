import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.gather import GatherNodesOutgoing, GatherState, GatherNodes
from kgcnn.layers.keras import Dense, Activation, Add, Multiply, Concatenate
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingWeightedLocalEdges, PoolingGlobalEdges, \
    PoolingNodes
from kgcnn.ops.activ import kgcnn_custom_act
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type


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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
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
                 node_indexing='sample',
                 pooling_method='sum',
                 is_sorted=False,
                 has_unconnected=True,
                 normalize_by_weights=False,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.normalize_by_weights = normalize_by_weights
        self.partition_type = partition_type
        self.pooling_method = pooling_method
        self.has_unconnected = has_unconnected
        self.is_sorted = is_sorted
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

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
        self.lay_gather = GatherNodesOutgoing(node_indexing=self.node_indexing, partition_type=self.partition_type,
                                              is_sorted=self.is_sorted, has_unconnected=self.has_unconnected,
                                              ragged_validate=self.ragged_validate,
                                              input_tensor_type=self.input_tensor_type)
        self.lay_dense = Dense(units=self.units, use_bias=self.use_bias, activation='linear',
                               input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                               **kernel_args)
        self.lay_pool = PoolingWeightedLocalEdges(pooling_method=self.pooling_method, is_sorted=self.is_sorted,
                                                  has_unconnected=self.has_unconnected,
                                                  node_indexing=self.node_indexing,
                                                  normalize_by_weights=self.normalize_by_weights,
                                                  partition_type=self.partition_type,
                                                  ragged_validate=self.ragged_validate,
                                                  input_tensor_type=self.input_tensor_type)
        self.lay_act = Activation(activation, ragged_validate=self.ragged_validate,
                                  input_tensor_type=self.input_tensor_type)

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

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
            embeddings: Node embeddings.
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
        config.update({"node_indexing": self.node_indexing,
                       "normalize_by_weights": self.normalize_by_weights,
                       "pooling_method": self.pooling_method,
                       "has_unconnected": self.has_unconnected,
                       "is_sorted": self.is_sorted,
                       "partition_type": self.partition_type,
                       "units": self.units,
                       "use_bias": self.use_bias,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate,
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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
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
                 node_indexing='sample',
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.is_sorted = is_sorted
        self.partition_type = partition_type
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

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
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias,
                                input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                **kernel_args)
        self.lay_dense2 = Dense(units=self.units, activation='linear', use_bias=self.use_bias,
                                input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                **kernel_args)
        self.lay_sum = PoolingLocalEdges(pooling_method=self.cfconv_pool,
                                         is_sorted=self.is_sorted, has_unconnected=self.has_unconnected,
                                         partition_type=self.partition_type, node_indexing=self.node_indexing,
                                         input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate)
        self.gather_n = GatherNodesOutgoing(node_indexing=self.node_indexing, partition_type=self.partition_type,
                                            input_tensor_type=self.input_tensor_type, is_sorted=self.is_sorted,
                                            ragged_validate=self.ragged_validate, has_unconnected=self.has_unconnected
                                            )
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
        config.update({"cfconv_pool": self.cfconv_pool,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "use_bias": self.use_bias,
                       "units": self.units,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate,
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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
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
                 cfconv_pool='sum',
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='sample',
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.partition_type = partition_type
        self.node_indexing = node_indexing
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

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
                                       input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                                       node_indexing=self.node_indexing, **kernel_args)
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
        config.update({"cfconv_pool": self.cfconv_pool,
                       "is_sorted}": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "node_indexing": self.node_indexing,
                       "units": self.units,
                       "use_bias": self.use_bias,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate,
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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
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
                 pooling_method="mean",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='sample',
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(MEGnetBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.partition_type = partition_type
        self.node_indexing = node_indexing
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

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
        mlp_args = {"input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate}
        mlp_args.update(kernel_args)
        pool_args = {"pooling_method": self.pooling_method, "is_sorted": self.is_sorted,
                     "has_unconnected": self.has_unconnected, "input_tensor_type": self.input_tensor_type,
                     "ragged_validate": self.ragged_validate, "partition_type": self.partition_type,
                     "node_indexing": self.node_indexing}
        gather_args = {"is_sorted": self.is_sorted, "has_unconnected": self.has_unconnected,
                       "input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate,
                       "partition_type": self.partition_type, "node_indexing": self.node_indexing}
        # Node
        self.lay_phi_n = Dense(units=self.node_embed[0], activation=activation, use_bias=self.use_bias, **mlp_args)
        self.lay_phi_n_1 = Dense(units=self.node_embed[1], activation=activation, use_bias=self.use_bias, **mlp_args)
        self.lay_phi_n_2 = Dense(units=self.node_embed[2], activation='linear', use_bias=self.use_bias, **mlp_args)
        self.lay_esum = PoolingLocalEdges(**pool_args)
        self.lay_gather_un = GatherState(**gather_args)
        self.lay_conc_nu = Concatenate(axis=-1, input_tensor_type=self.input_tensor_type)
        # Edge
        self.lay_phi_e = Dense(units=self.edge_embed[0], activation=activation, use_bias=self.use_bias, **mlp_args)
        self.lay_phi_e_1 = Dense(units=self.edge_embed[1], activation=activation, use_bias=self.use_bias, **mlp_args)
        self.lay_phi_e_2 = Dense(units=self.edge_embed[2], activation='linear', use_bias=self.use_bias, **mlp_args)
        self.lay_gather_n = GatherNodes(**gather_args)
        self.lay_gather_ue = GatherState(**gather_args)
        self.lay_conc_enu = Concatenate(axis=-1, input_tensor_type=self.input_tensor_type)
        # Environment
        self.lay_usum_e = PoolingGlobalEdges(**pool_args)
        self.lay_usum_n = PoolingNodes(**pool_args)
        self.lay_conc_u = Concatenate(axis=-1, input_tensor_type="tensor")
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

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs: [nodes, edges, edge_index, state]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edges: Edge or message embeddings of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)
            - state (tf.tensor): State information for the graph, a single tensor of shape (batch, F)

        Returns:
            node_update: Updated node embeddings.
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
        config.update({"is_sorted": self.is_sorted,
                       "pooling_method": self.pooling_method,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "node_indexing": self.node_indexing,
                       "node_embed": self.node_embed,
                       "edge_embed": self.edge_embed,
                       "env_embed": self.env_embed,
                       "use_bias": self.use_bias,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate,
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


class DimNetOutputBlock(ks.layers.Layer):
    """
    Megnet Block.

    Args:
        emb_size (list, optional): List of node embedding dimension. Defaults to [16,16,16].
        out_emb_size (list, optional): List of edge embedding dimension. Defaults to [16,16,16].
        num_dense (list, optional): List of environment embedding dimension. Defaults to [16,16,16].
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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self, emb_size,
                 out_emb_size,
                 num_dense,
                 num_targets=12,
                 output_kernel_initializer="zeros",
                 kernel_initializer='orthogonal',
                 bias_initializer='zeros',
                 use_bias=True,
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 pooling_method="sum",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 node_indexing='sample',
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(DimNetOutputBlock, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.partition_type = partition_type
        self.node_indexing = node_indexing
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size
        self.num_dense = num_dense
        self.num_targets = num_targets

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

        self.use_bias = use_bias
        if activation is None and 'swish' in kgcnn_custom_act:
            activation = 'swish'
        elif activation is None:
            activation = "selu"

        self.dense_output_kernel_initializer = tf.keras.initializers.get(output_kernel_initializer)
        self.dense_activation = tf.keras.activations.get(activation)
        self.dense_kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.dense_bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.dense_kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.dense_bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.dense_activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.dense_kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.dense_bias_constraint = tf.keras.constraints.get(bias_constraint)

        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_initializer": bias_initializer}
        mlp_args = {"input_tensor_type": self.input_tensor_type, "ragged_validate": self.ragged_validate,
                    "bias_regularizer": bias_regularizer, "bias_constraint": bias_constraint,
                    "bias_initializer": bias_initializer}
        mlp_args.update(kernel_args)
        pool_args = {"pooling_method": self.pooling_method, "is_sorted": self.is_sorted,
                     "has_unconnected": self.has_unconnected, "input_tensor_type": self.input_tensor_type,
                     "ragged_validate": self.ragged_validate, "partition_type": self.partition_type,
                     "node_indexing": self.node_indexing}

        self.dense_rbf = Dense(emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.up_projection = Dense(out_emb_size, use_bias=False, kernel_initializer=kernel_initializer, **kernel_args)
        self.dense_mlp = MLP([out_emb_size] * num_dense, activation=activation, use_bias=use_bias, **mlp_args)
        self.dimnet_mult = Multiply(input_tensor_type=self.input_tensor_type)
        self.pool = PoolingLocalEdges(**pool_args)
        self.dense_final = Dense(num_targets, use_bias=False, kernel_initializer=output_kernel_initializer,
                                 **kernel_args)

    def build(self, input_shape):
        """Build layer."""
        super(DimNetOutputBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs: [nodes, edges, edge_index, state]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edges: Edge or message embeddings of shape (batch, [M], F)
            - rbf: Edge distance basis of shape (batch, [M], F)
            - edge_index: Node indices of shape (batch, [M], 2)

        Returns:
            node_update: Updated node embeddings.
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
        config.update({"is_sorted": self.is_sorted,
                       "pooling_method": self.pooling_method,
                       "has_unconnected": self.has_unconnected,
                       "partition_type": self.partition_type,
                       "node_indexing": self.node_indexing,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate,
                       })
        config.update({"use_bias": self.use_bias,
                       "activation": tf.keras.activations.serialize(self.dense_activation),
                       "kernel_regularizer": tf.keras.regularizers.serialize(self.dense_kernel_regularizer),
                       "bias_regularizer": tf.keras.regularizers.serialize(self.dense_bias_regularizer),
                       "activity_regularizer": tf.keras.regularizers.serialize(self.dense_activity_regularizer),
                       "kernel_constraint": tf.keras.constraints.serialize(self.dense_kernel_constraint),
                       "bias_constraint": tf.keras.constraints.serialize(self.dense_bias_constraint),
                       "kernel_initializer": tf.keras.initializers.serialize(self.dense_kernel_initializer),
                       "bias_initializer": tf.keras.initializers.serialize(self.dense_bias_initializer)
                       })
        config.update({"emb_size": self.emb_size,
                       "out_emb_size": self.out_emb_size,
                       "num_dense": self.num_dense,
                       "num_targets": self.num_targets,
                       "output_kernel_initializer": tf.keras.initializers.serialize(
                           self.dense_output_kernel_initializer),
                       })
        return config


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 name='residual', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_1 = Dense(units, activation=activation, use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer)
        self.dense_2 = Dense(units, activation=activation, use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer)
        self.add_end = Add()

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.add_end([inputs, x])
        return x


class DimNetInteractionPPBlock(tf.keras.layers.Layer):

    def __init__(self, emb_size,
                 int_emb_size,
                 basis_emb_size,
                 num_before_skip,
                 num_after_skip,
                 activation=None,
                 kernel_initializer="orthogonal",
                 **kwargs):
        super(DimNetInteractionPPBlock, self).__init__(**kwargs)

        if activation is None and 'swish' in kgcnn_custom_act:
            activation = 'swish'
        elif activation is None:
            activation = "selu"

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = Dense(basis_emb_size, use_bias=False, kernel_initializer=kernel_initializer)
        self.dense_rbf2 = Dense(emb_size, use_bias=False, kernel_initializer=kernel_initializer)
        self.dense_sbf1 = Dense(basis_emb_size, use_bias=False, kernel_initializer=kernel_initializer)
        self.dense_sbf2 = Dense(int_emb_size, use_bias=False, kernel_initializer=kernel_initializer)

        # Dense transformations of input messages
        self.dense_ji = Dense(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=kernel_initializer)
        self.dense_kj = Dense(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=kernel_initializer)

        # Embedding projections for interaction triplets
        self.down_projection = Dense(int_emb_size, activation=activation, use_bias=False,
                                     kernel_initializer=kernel_initializer)
        self.up_projection = Dense(emb_size, activation=activation, use_bias=False,
                                   kernel_initializer=kernel_initializer)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=kernel_initializer))
        self.final_before_skip = Dense(emb_size, activation=activation, use_bias=True,
                                       kernel_initializer=kernel_initializer)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=kernel_initializer))

        self.lay_add1 = Add()
        self.lay_add2 = Add()
        self.lay_mult1 = Multiply()
        self.lay_mult2 = Multiply()

        self.lay_gather = GatherNodesOutgoing()
        self.lay_pool = PoolingLocalEdges()

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
