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

