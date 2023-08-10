import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.segment import segment_ops_by_name, segment_softmax

# For backward compatibility
# noinspection PyUnresolvedReferences
from kgcnn.layers.aggr import (
    AggregateLocalEdges,
    AggregateLocalMessages,
    PoolingLocalEdges,
    PoolingLocalMessages,
    AggregateWeightedLocalEdges,
    PoolingWeightedLocalEdges,
    PoolingWeightedLocalMessages,
    AggregateWeightedLocalMessages,
    AggregateLocalEdgesLSTM,
    PoolingLocalMessagesLSTM,
    PoolingLocalEdgesLSTM,
    AggregateLocalEdgesAttention,
    PoolingLocalEdgesAttention,
    AggregateLocalMessagesAttention,
    AggregateLocalMessagesLSTM,
    RelationalPoolingLocalEdges,
    RelationalAggregateLocalMessages,
    RelationalAggregateLocalEdges
)

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingEmbedding')
class PoolingEmbedding(GraphBaseLayer):
    r"""Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    :obj:`tf.Tensor` .

    """

    def __init__(self, pooling_method: str = "mean", **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        """
        super(PoolingEmbedding, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.node_indexing = "sample"

    def build(self, input_shape):
        """Build layer."""
        super(PoolingEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Embedding tensor of shape (batch, [N], F)

        Returns:
            tf.Tensor: Pooled node features of shape (batch, F)
        """
        # Need ragged input but can be generalized in the future.
        inputs = self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, batchi = inputs.values, inputs.value_rowids()

        # Could also use reduce_sum here.
        out = segment_ops_by_name(self.pooling_method, nod, batchi)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingEmbedding, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


PoolingNodes = PoolingEmbedding
PoolingGlobalEdges = PoolingEmbedding


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingWeightedEmbedding')
class PoolingWeightedEmbedding(GraphBaseLayer):
    r"""Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    :obj:`tf.Tensor` .

    .. note::

        In addition to pooling embeddings a weight tensor must be supplied that scales each embedding before
        pooling. Must broadcast.

    """

    def __init__(self, pooling_method: str = "mean", **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        """
        super(PoolingWeightedEmbedding, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.node_indexing = "sample"

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, weights]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - weights (tf.RaggedTensor): Node or message weights. Most broadcast to nodes. Shape (batch, [N], 1).

        Returns:
            tf.Tensor: Pooled node features of shape (batch, F)
        """
        # Need ragged input but can be generalized in the future.
        inputs = self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, batchi = inputs[0].values, inputs[0].value_rowids()
        weights, _ = inputs[1].values, inputs[1].value_rowids()
        nod = tf.math.multiply(nod, weights)
        # Could also use reduce_sum here.
        out = segment_ops_by_name(self.pooling_method, nod, batchi)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedEmbedding, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config


PoolingWeightedNodes = PoolingWeightedEmbedding
PoolingWeightedGlobalEdges = PoolingWeightedEmbedding


@ks.utils.register_keras_serializable(package='kgcnn', name='PoolingEmbeddingAttention')
class PoolingEmbeddingAttention(GraphBaseLayer):
    r"""Polling all embeddings of edges or nodes per batch to obtain a graph level embedding in form of a
    ::obj`tf.Tensor` .

    Uses attention for pooling. i.e. :math:`s =  \sum_j \alpha_{i} n_i` .
    The attention is computed via: :math:`\alpha_i = \text{softmax}_i(a_i)` from the attention
    coefficients :math:`a_i` .
    The attention coefficients must be computed beforehand by edge features or by :math:`\sigma( W [s || n_i])` and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, :math:`s =  \sum_i \text{softmax}_j(a_i) n_i` is computed by the layer.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(PoolingEmbeddingAttention, self).__init__(**kwargs)
        self.node_indexing = "sample"

    def build(self, input_shape):
        """Build layer."""
        super(PoolingEmbeddingAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, attention]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - attention (tf.RaggedTensor): Attention coefficients of shape (batch, [N], 1)

        Returns:
            tf.Tensor: Embedding tensor of pooled node of shape (batch, F)
        """
        # Need ragged input but can be generalized in the future.
        inputs = self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, batchi, target_len = inputs[0].values, inputs[0].value_rowids(), inputs[0].row_lengths()
        ats = inputs[1].values

        ats = segment_softmax(ats, batchi)
        get = nod * ats
        out = tf.math.segment_sum(get, batchi)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingEmbeddingAttention, self).get_config()
        return config


PoolingNodesAttention = PoolingEmbeddingAttention
