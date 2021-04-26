import tensorflow as tf
import tensorflow.keras as ks

# from kgcnn.layers.ragged.pooling import PoolingLocalEdges,PoolingNodes
from kgcnn.layers.ragged.gather import GatherNodesOutgoing
from kgcnn.layers.ragged.pooling import PoolingWeightedLocalEdges, PoolingLocalEdges
# from kgcnn.layers.ragged.gather import GatherState,GatherNodesIngoing
from kgcnn.ops.activ import kgcnn_custom_act


# import tensorflow.keras.backend as ksb
class DenseRagged(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer same as Dense."""
        super(DenseRagged, self).__init__(**kwargs)
        self.ragged_dense = ks.layers.Dense(units=units, activation=activation,
                                            use_bias=use_bias)
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        value_tensor = inputs.values
        out_tensor = self.ragged_dense(value_tensor)
        return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)


# class DenseRagged(tf.keras.layers.Layer):
#     """
#     Custom Dense Layer for ragged input. The dense layer can be used as convolutional unit.
#     This is a placeholder until Dense supports ragged tensors.
#
#     Arguments:
#         units: Positive integer, dimensionality of the output space.
#         activation: Activation function to use. If you don't specify anything, no activation is applied
#                     (ie. "linear" activation: `a(x) = x`).
#         use_bias: Boolean, whether the layer uses a bias vector.
#         kernel_initializer: Initializer for the `kernel` weights matrix.
#         bias_initializer: Initializer for the bias vector.
#         kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
#         bias_regularizer: Regularizer function applied to the bias vector.
#         activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
#         kernel_constraint: Constraint function applied to the `kernel` weights matrix.
#         bias_constraint: Constraint function applied to the bias vector.
#
#     Input shape:
#         N-D tensor with shape: `(batch_size, ..., input_dim)`.
#         The most common situation would be
#         a 2D input with shape `(batch_size, input_dim)`.
#
#     Output shape:
#         N-D tensor with shape: `(batch_size, ..., units)`.
#         For instance, for a 2D input with shape `(batch_size, input_dim)`,
#         the output would have shape `(batch_size, units)`.
#     """
#
#     def __init__(self,
#                  units,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         """Initialize layer same as Dense."""
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#
#         super(DenseRagged, self).__init__(**kwargs)
#
#         self.units = int(units) if not isinstance(units, int) else units
#         self.activation = ks.activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = ks.initializers.get(kernel_initializer)
#         self.bias_initializer = ks.initializers.get(bias_initializer)
#         self.kernel_regularizer = ks.regularizers.get(kernel_regularizer)
#         self.bias_regularizer = ks.regularizers.get(bias_regularizer)
#         self.kernel_constraint = ks.constraints.get(kernel_constraint)
#         self.bias_constraint = ks.constraints.get(bias_constraint)
#
#         self._supports_ragged_inputs = True
#         self.kernel = None
#         self.bias = None
#
#     def build(self, input_shape):
#         """Build layer's kernel and bias."""
#         last_dim = input_shape[-1]
#
#         # Add Kernel
#         self.kernel = self.add_weight('kernel',
#                                       shape=[last_dim, self.units],
#                                       initializer=self.kernel_initializer,
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint,
#                                       dtype=self.dtype,
#                                       trainable=True)
#         # Add bias
#         if self.use_bias:
#             self.bias = self.add_weight('bias',
#                                         shape=[self.units, ],
#                                         initializer=self.bias_initializer,
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint,
#                                         dtype=self.dtype,
#                                         trainable=True)
#         else:
#             self.bias = None
#
#         super(DenseRagged, self).build(input_shape)  # should set sef.built = True
#
#     def call(self, inputs, **kwargs):
#         """Forward pass."""
#         outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
#         if self.use_bias:
#             outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
#
#         outputs = tf.ragged.map_flat_values(self.activation, outputs)
#         return outputs
#
#     def get_config(self):
#         """Update config."""
#         config = super(DenseRagged, self).get_config()
#         config.update({
#             'units':
#                 self.units,
#             'activation':
#                 ks.activations.serialize(self.activation),
#             'use_bias':
#                 self.use_bias,
#             'kernel_initializer':
#                 ks.initializers.serialize(self.kernel_initializer),
#             'bias_initializer':
#                 ks.initializers.serialize(self.bias_initializer),
#             'kernel_regularizer':
#                 ks.regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer':
#                 ks.regularizers.serialize(self.bias_regularizer),
#             'kernel_constraint':
#                 ks.constraints.serialize(self.kernel_constraint),
#             'bias_constraint':
#                 ks.constraints.serialize(self.bias_constraint)
#         })
#         return config


class ActivationRagged(tf.keras.layers.Layer):
    """
    Applies an activation function to an output.
    
    Arguments:
        activation: Activation function, such as `tf.nn.relu`, or string name of built-in.
    """

    def __init__(self, activation, **kwargs):
        """Initialize layer same as Activation."""
        super(ActivationRagged, self).__init__(**kwargs)
        self.activation = ks.activations.get(activation)
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        """Forward pass.
        
        Args:
            tensor (tf.ragged): Ragged tensor of shape e.g. (batch,None,F)
        
        Returns:
            tf.ragged: Elementwise activation of flat values.
        """
        out = tf.ragged.map_flat_values(self.activation, inputs)
        return out

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Update config."""
        base_config = super(ActivationRagged, self).get_config()
        config = {'activation': ks.activations.serialize(self.activation)}
        config.update(base_config)
        return config


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
        units (int): Output dimension / units of dense layer.
        use_bias (bool): Use bias. Default is True.
        activation (str): Activation. Default is {"class_name": "leaky_relu", "config": {"alpha": 0.2}},
            with fall-back "relu".
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        pooling_method (str): Pooling method for summing edges 'segment_sum'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        **kwargs
    """

    def __init__(self,
                 units,
                 use_bias=False,
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 node_indexing="sample",
                 pooling_method='segment_sum',
                 is_sorted=False,
                 has_unconnected=True,
                 normalize_by_weights=False,
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.normalize_by_weights = normalize_by_weights
        self.pooling_method = pooling_method
        self.has_unconnected = has_unconnected
        self.is_sorted = is_sorted
        self.ragged_validate = ragged_validate

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
        self.lay_gather = GatherNodesOutgoing(node_indexing=self.node_indexing, ragged_validate=self.ragged_validate)
        self.lay_dense = DenseRagged(units=self.units, use_bias=self.use_bias, activation='linear',
                                     **kernel_args)
        self.lay_pool = PoolingWeightedLocalEdges(pooling_method=self.pooling_method, is_sorted=self.is_sorted,
                                                  has_unconnected=self.has_unconnected,
                                                  node_indexing=self.node_indexing,
                                                  normalize_by_weights=self.normalize_by_weights,
                                                  ragged_validate=self.ragged_validate)
        self.lay_act = ActivationRagged(activation)
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edge, edge_index]
        
        Inputs:
            nodes (tf.ragged): Ragged node feature list of shape (batch,None,F)
            edges (tf.ragged): Ragged edge feature list of shape (batch,None,F)
            edge_index (tf.ragged): Edge indices for (batch,None,2)
        
        Returns:
            features (tf.ragged): adj_matrix list of updated node features.
            Output shape is (batch,None,F).
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
                       "ragged_validate": self.ragged_validate,
                       "is_sorted": self.is_sorted,
                       "use_bias": self.use_bias,
                       "units": self.units,
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
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
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
                 node_indexing='sample',
                 ragged_validate=False,
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.ragged_validate = ragged_validate

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
        self.lay_dense1 = DenseRagged(units=self.units, activation=activation, use_bias=self.use_bias,
                                      **kernel_args)
        self.lay_dense2 = DenseRagged(units=self.units, activation='linear', use_bias=self.use_bias,
                                      **kernel_args)
        self.lay_sum = PoolingLocalEdges(pooling_method=self.cfconv_pool,
                                         is_sorted=self.is_sorted,
                                         has_unconnected=self.has_unconnected,
                                         node_indexing=self.node_indexing,
                                         ragged_validate=self.ragged_validate)
        self.gather_n = GatherNodesOutgoing(node_indexing=self.node_indexing, ragged_validate=self.ragged_validate)
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate edge update.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.

        Returns:
            node_update (tf.tensor): Updated node features of shape (batch*None,F)
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(edge)
        x = self.lay_dense2(x)
        node2exp = self.gather_n([node, indexlist])
        x = node2exp * x
        x = self.lay_sum([node, x, indexlist])
        return x

    def get_config(self):
        """Update layer config."""
        config = super(SchNetCFconv, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate,
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
                 node_indexing='sample',
                 ragged_validate=False,
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.ragged_validate = ragged_validate
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
                                       is_sorted=self.is_sorted, node_indexing=self.node_indexing,
                                       ragged_validate=self.ragged_validate,
                                       **kernel_args)
        self.lay_dense1 = DenseRagged(units=self.units, activation='linear', use_bias=False, **kernel_args)
        self.lay_dense2 = DenseRagged(units=self.units, activation=activation, use_bias=self.use_bias,
                                      **kernel_args)
        self.lay_dense3 = DenseRagged(units=self.units, activation='linear', use_bias=self.use_bias,
                                      **kernel_args)
        self.lay_add = ks.layers.Add()
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass: Calculate node update.

        Args:
            inputs (list): [node, node_partition, edge, edge_partition, edge_index]

            - nodes (tf.tensor): Flatten node feature list of shape (batch*None,F)
            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_index (tf.tensor): Edge indices for disjoint representation of shape
              (batch*None,2) that corresponds to indexing 'batch'.

        Returns:
            node_update (tf.tensor): Updated node features.
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
                       "node_indexing": self.node_indexing,
                       "ragged_validate": self.ragged_validate,
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
