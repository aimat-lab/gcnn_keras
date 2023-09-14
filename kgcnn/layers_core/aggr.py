import keras_core as ks
import keras_core.saving
from keras_core.layers import Layer
from keras_core import ops
from kgcnn.ops_core.scatter import (
    scatter_reduce_min, scatter_reduce_mean, scatter_reduce_max, scatter_reduce_sum, scatter_reduce_softmax)


@ks.saving.register_keras_serializable(package='kgcnn', name='Aggregate')
class Aggregate(Layer):  # noqa

    def __init__(self, pooling_method: str = "scatter_sum", axis=0, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.axis = axis
        if axis != 0:
            raise NotImplementedError()
        pooling_by_name = {
            "scatter_sum": scatter_reduce_sum,
            "scatter_mean": scatter_reduce_mean,
            "scatter_max": scatter_reduce_max,
            "scatter_min": scatter_reduce_min,
            "segment_sum": None,
            "segment_mean": None,
            "segment_max": None,
            "segment_min": None
        }
        self._pool_method = pooling_by_name[pooling_method]
        self._use_scatter = "scatter" in pooling_method
        
    def build(self, input_shape):
        # Nothing to build here. No sub-layers.
        self.built = True
        
    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        x_shape, _, dim_size = input_shape
        return tuple(list(dim_size[:1]) + list(x_shape[1:]))

    def call(self, inputs, **kwargs):
        x, index, reference = inputs
        shape = ops.shape(reference)[:1] + ops.shape(x)[1:]
        if self._use_scatter:
            return self._pool_method(index, x, shape=shape)
        else:
            raise NotImplementedError()


class AggregateLocalEdges(Layer):

    def __init__(self, pooling_method="scatter_sum", pooling_index: int = 1, **kwargs):
        super(AggregateLocalEdges, self).__init__(**kwargs)
        self.pooling_index = pooling_index
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        assert len(input_shape) == 3
        node_shape, edges_shape, edge_index_shape = input_shape
        self.to_aggregate.build((edges_shape, edge_index_shape[1:], node_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        node_shape, edges_shape, edge_index_shape = input_shape
        return self.to_aggregate.compute_output_shape([edges_shape, edge_index_shape[1:], node_shape])

    def call(self, inputs, **kwargs):
        n, edges, edge_index = inputs
        return self.to_aggregate([edges, edge_index[self.pooling_index], n])


class AggregateWeightedLocalEdges(AggregateLocalEdges):

    def __init__(self, pooling_method="scatter_sum", pooling_index: int = 1, normalize_by_weights=False, **kwargs):
        super(AggregateWeightedLocalEdges, self).__init__(**kwargs)
        self.normalize_by_weights = normalize_by_weights
        self.pooling_index = pooling_index
        self.to_aggregate = Aggregate(pooling_method=pooling_method)
        self.to_aggregate_weights = Aggregate(pooling_method="scatter_sum")

    def build(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, edge_index_shape, weights_shape = input_shape
        self.to_aggregate.build((edges_shape, edge_index_shape[1:], node_shape))
        self.to_aggregate_weights.build((weights_shape, edge_index_shape[1:], node_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, edge_index_shape, weights_shape = input_shape
        return self.to_aggregate.compute_output_shape([edges_shape, edge_index_shape[1:], node_shape])

    def call(self, inputs, **kwargs):
        n, edges, edge_index, weights = inputs
        edges = edges*weights

        out = self.to_aggregate([edges, edge_index[self.pooling_index], n])

        if self.normalize_by_weights:
            norm = self.to_aggregate_weights([weights, edge_index[self.pooling_index], n])
            out = out/norm
        return out


class AggregateLocalEdgesAttention(Layer):
    r"""Aggregate local edges via Attention mechanism.
    Uses attention for pooling. i.e. :math:`n_i =  \sum_j \alpha_{ij} e_{ij}`
    The attention is computed via: :math:`\alpha_ij = \text{softmax}_j (a_{ij})` from the
    attention coefficients :math:`a_{ij}` .
    The attention coefficients must be computed beforehand by edge features or by :math:`\sigma( W n_i || W n_j)` and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, :math:`n_i = \sum_j \text{softmax}_j (a_{ij}) e_{ij}` is computed by the layer.
    """

    def __init__(self,
                 softmax_method="scatter_softmax",
                 pooling_method="scatter_sum",
                 pooling_index: int = 1,
                 is_sorted: bool = False,
                 has_unconnected: bool = True,
                 normalize: bool = False,
                 **kwargs):
        """Initialize layer.

        Args:
            pooling_method (str): Pooling method for this layer.
            pooling_index (int): Index to pick ID's for pooling edge-like embeddings. Default is 0.
            is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
            has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        """
        super(AggregateLocalEdgesAttention, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.pooling_index = pooling_index
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.normalize = normalize
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, attention_shape, edge_index_shape = input_shape
        self.to_aggregate.build((edges_shape, edge_index_shape[1:], node_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        node_shape, edges_shape, attention_shape, edge_index_shape = input_shape
        return self.to_aggregate.compute_output_shape([edges_shape, edge_index_shape[1:], node_shape])

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [node, edges, attention, edge_indices]

                - nodes (Tensor): Node embeddings of shape (N, F)
                - edges (Tensor): Edge or message embeddings of shape (M, F)
                - attention (Tensor): Attention coefficients of shape (M, 1)
                - edge_indices (Tensor): Edge indices referring to nodes of shape (2, M)

        Returns:
            Tensor: Embedding tensor of aggregated edge attentions for each node of shape (N, F)
        """
        reference, x, attention, disjoint_indices = inputs
        receive_indices = disjoint_indices[self.pooling_index]
        shape_attention = ops.shape(reference)[:1] + ops.shape(attention)[1:]
        a = scatter_reduce_softmax(receive_indices, attention, shape=shape_attention, normalize=self.normalize)
        x = x * ops.broadcast_to(a, ops.shape(x))
        return self.to_aggregate([x, receive_indices, reference])

    def get_config(self):
        """Update layer config."""
        config = super(AggregateLocalEdgesAttention, self).get_config()
        return config
