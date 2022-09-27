import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingWeightedLocalEdges
from kgcnn.layers.modules import ActivationEmbedding, DenseEmbedding


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GCN')
class GCN(GraphBaseLayer):
    r"""Graph convolution according to `Kipf et al <https://arxiv.org/abs/1609.02907>`_ .

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
        self.lay_gather = GatherNodesOutgoing()
        self.lay_dense = DenseEmbedding(units=self.units, activation='linear', **kernel_args)
        self.lay_pool = PoolingWeightedLocalEdges(**pool_args)
        self.lay_act = ActivationEmbedding(activation)

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
        no = self.lay_dense(node, **kwargs)
        no = self.lay_gather([no, edge_index], **kwargs)
        nu = self.lay_pool([node, no, edge_index, edges], **kwargs)  # Summing for each node connection
        out = self.lay_act(nu, **kwargs)
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


class GCNConvPoolingSparseAdjacencyMatmul(GraphBaseLayer):
    r"""Layer for graph convolution of node embeddings by multiplying with sparse adjacency matrix, as proposed
    in Graph convolution according to `Kipf et al <https://arxiv.org/abs/1609.02907>`_ .

    :math:`A x`, where :math:`A` represents the possibly scaled adjacency matrix.

    The node features are flatten for a disjoint representation.

    Args:
        pooling_method (str): Not used. Default is "sum".
    """

    def __init__(self, pooling_method="sum", **kwargs):
        """Initialize layer."""
        super(GCNConvPoolingSparseAdjacencyMatmul, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(GCNConvPoolingSparseAdjacencyMatmul, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, adjacency]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - adjacency (tf.SparseTensor): SparseTensor of the adjacency matrix of shape (batch*[N], batch*[N])

        Returns:
            tf.RaggedTensor: Pooled node features of shape (batch, [N], F)
        """
        inputs = self.assert_ragged_input_rank(inputs[0])
        adj = inputs[1]
        node, node_part = inputs[0].values, inputs[0].row_splits
        out = tf.sparse.sparse_dense_matmul(adj, node)
        out = tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(GCNConvPoolingSparseAdjacencyMatmul, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config