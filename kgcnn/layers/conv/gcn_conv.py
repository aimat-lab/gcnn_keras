import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pool.pooling import PoolingWeightedLocalEdges
from kgcnn.layers.keras import Activation, Dense


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GCN')
class GCN(GraphBaseLayer):
    r"""Graph convolution according to Kipf et al.

    Computes graph convolution as :math:`\sigma(A_s(WX+b))` where :math:`A_s` is the precomputed and scaled adjacency
    matrix. The scaled adjacency matrix is defined by :math:`A_s = D^{-0.5} (A + I) D^{-0.5}` with the degree
    matrix :math:`D`. In place of :math:`A_s`, this layers uses edge features (that are the entries of :math:`A_s`) and
    edge indices. :math:`A_s` is considered pre-scaled, this is not done by this layer.
    If no scaled edge features are available, you could consider use e.g. "segment_mean", or normalize_by_weights to
    obtain a similar behaviour that is expected by a pre-scaled adjacency matrix input.
    Edge features must be possible to broadcast to node features. Ideally they have shape (..., 1).

    Args:
        units (int): Output dimension/ units of dense layer.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
            In this case the edge features are considered weights of dimension (...,1) and are summed for each node.
        activation (str): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}}.
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
                 activation='kgcnn>leaky_relu',
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
        self.lay_gather = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_dense = Dense(units=self.units, activation='linear',
                               input_tensor_type=self.input_tensor_type, ragged_validate=self.ragged_validate,
                               **kernel_args)
        self.lay_pool = PoolingWeightedLocalEdges(**pool_args, **self._kgcnn_info)
        self.lay_act = Activation(activation, ragged_validate=self.ragged_validate,
                                  input_tensor_type=self.input_tensor_type)

    def build(self, input_shape):
        """Build layer."""
        super(GCN, self).build(input_shape)

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