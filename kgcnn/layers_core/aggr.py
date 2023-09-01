from keras_core.layers import Layer
from keras_core import ops


# @keras_core.saving.register_keras_serializable()
class Aggregate(Layer):

    def __init__(self, pooling_method: str = "sum", axis=0, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.axis = axis
        if axis != 0:
            raise NotImplementedError

    def call(self, inputs, **kwargs):
        x, index, dim_size = inputs
        # For test only sum scatter, no segment operation no other poolings etc.
        # will add all poolings here.
        shape = ops.concatenate([dim_size, ops.shape(x)[1:]])
        return ops.scatter(ops.expand_dims(index, axis=-1), x, shape=shape)


class AggregateLocalEdges(Layer):

    def __init__(self, pooling_method="sum", pooling_index: int = 1, **kwargs):
        super(AggregateLocalEdges, self).__init__(**kwargs)
        self.pooling_index = pooling_index
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        self.to_aggregate.build((input_shape[0], input_shape[1][1:], input_shape[0][:1]))

    def call(self, inputs, **kwargs):
        n, edges, edge_index = inputs
        # For test only sum scatter, no segment operation etc.
        return self.to_aggregate([edges, edge_index[self.pooling_index], ops.cast(ops.shape(n)[:1], dtype="int64")])


class AggregateWeightedLocalEdges(AggregateLocalEdges):

    def __init__(self, pooling_method="sum", pooling_index: int = 1, normalize_by_weights=False, **kwargs):
        super(AggregateWeightedLocalEdges, self).__init__(**kwargs)
        self.normalize_by_weights = normalize_by_weights
        self.pooling_index = pooling_index
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        self.to_aggregate.build((input_shape[0], input_shape[1][1:], input_shape[0][:1]))

    def call(self, inputs, **kwargs):
        n, edges, edge_index, weights = inputs
        edges = edges*weights
        # For test only sum scatter, no segment operation etc.
        out = self.to_aggregate([edges, edge_index[self.pooling_index], ops.cast(ops.shape(n)[:1], dtype="int64")])

        # if self.normalize_by_weights:
        #     norm = tensor_scatter_nd_ops_by_name(
        #         "add", tf.zeros(tf.concat([tf.shape(nodes)[:1], tf.shape(edges)[1:]], axis=0), dtype=edges.dtype),
        #         tf.expand_dims(receive_indices, axis=-1), weights
        #     )
        #     # We could also optionally add tf.eps here.
        #     out = tf.math.divide_no_nan(out, norm)
        return out
