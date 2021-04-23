import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.sparse.pooling import PoolingAdjacencyMatmul
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
                 node_indexing="batch",
                 pooling_method='sum',
                 is_sorted=False,
                 has_unconnected=True,
                 normalize_by_weights=False,
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.normalize_by_weights = normalize_by_weights
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
        self.lay_dense = ks.layers.Dense(units=self.units, use_bias=self.use_bias, activation='linear', **kernel_args)
        self.mult_adj = PoolingAdjacencyMatmul()
        self.lay_act = ks.layers.Activation(activation)
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            list: [nodes, adjacency]

            - nodes (tf.tensor): Flatten node features of shape (batch*None,F)
            - adjacency (tf.sparse): SparseTensor of the adjacency matrix of shape (batch*None,batch*None)

        Returns:
            features (tf.tensor): Pooled node features of shape (batch,F)
        """
        nod, adj = inputs
        nod = self.lay_dense(nod)
        an = self.mult_adj([nod, adj])
        out = self.lay_act(an)
        return out

    def get_config(self):
        """Update config."""
        config = super(GCN, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "normalize_by_weights": self.normalize_by_weights,
                       "pooling_method": self.pooling_method,
                       "has_unconnected": self.has_unconnected,
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