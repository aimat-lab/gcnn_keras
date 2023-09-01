from keras_core.layers import Layer, Dense, Activation
from kgcnn.layers_core.aggr import AggregateWeightedLocalEdges
from kgcnn.layers_core.gather import GatherNodesOutgoing
from keras_core import ops


class GCN(Layer):
    r"""Graph convolution according to `Kipf et al <https://arxiv.org/abs/1609.02907>`__ .

    Computes graph convolution as :math:`\sigma(A_s(WX+b))` where :math:`A_s` is the precomputed and scaled adjacency
    matrix. The scaled adjacency matrix is defined by :math:`A_s = D^{-0.5} (A + I) D^{-0.5}` with the degree
    matrix :math:`D` . In place of :math:`A_s` , this layers uses edge features (that are the entries of :math:`A_s` )
    and edge indices.

    .. note::

        :math:`A_s` is considered pre-scaled, this is not done by this layer!
        If no scaled edge features are available, you could consider use e.g. "mean",
        or :obj:`normalize_by_weights` to obtain a similar behaviour that is expected b
        y a pre-scaled adjacency matrix input.

    Edge features must be possible to broadcast to node features, since they are multiplied with the node features.
    Ideally they are weights of shape `(..., 1)` for broadcasting, e.g. entries of :math:`A_s` .

    Args:
        units (int): Output dimension/ units of dense layer.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
            In this case the edge features are considered weights of dimension (...,1) and are summed for each node.
        activation (str): Activation. Default is 'kgcnn>leaky_relu'.
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
                 activation='relu',
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
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}
        pool_args = {"pooling_method": pooling_method, "normalize_by_weights": normalize_by_weights}

        # Layers
        self.layer_gather = GatherNodesOutgoing()
        self.layer_dense = Dense(units=self.units, activation='linear', **kernel_args)
        self.layer_pool = AggregateWeightedLocalEdges(**pool_args)
        self.layer_act = Activation(activation)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Node embeddings of shape (batch, [N], F)
        """
        node, edges, edge_index = inputs
        no = self.layer_dense(node, **kwargs)
        no = self.layer_gather([no, edge_index], **kwargs)
        nu = self.layer_pool([node, no, edge_index, edges], **kwargs)  # Summing for each node connection
        out = self.layer_act(nu, **kwargs)
        return out

    def get_config(self):
        """Update config."""
        config = super(GCN, self).get_config()
        config.update({"normalize_by_weights": self.normalize_by_weights,
                       "pooling_method": self.pooling_method, "units": self.units})
        conf_dense = self.layer_dense.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "use_bias"]:
            config.update({x: conf_dense[x]})
        conf_act = self.layer_act.get_config()
        config.update({"activation": conf_act["activation"]})
        return config
