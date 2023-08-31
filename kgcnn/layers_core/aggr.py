from keras_core.layers import Layer
from keras_core import ops

class Aggregate(Layer):

    def __init__(self, pooling_method="sum", axis=0, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.axis = axis
        if axis != 0:
            raise NotImplementedError

    def call(self, inputs, **kwargs):
        x, index, dim_size = inputs
        # For test only sum scatter, no segment operation no other poolings etc.
        # will add all poolings here.
        return ops.scatter(index, x, shape=ops.concatenate([[dim_size], ops.shape[1:]]))


class AggregateLocalEdges(Layer):
    def __init__(self, pooling_method="sum", **kwargs):
        super(AggregateLocalEdges, self).__init__(**kwargs)
        self.to_aggregate = Aggregate(pooling_method=pooling_method)

    def call(self, inputs, **kwargs):
        n, edges, edge_index = inputs
        # For test only sum scatter, no segment operation etc.
        return self.to_aggregate(edges, edge_index[1], ops.shape(n)[0])


class AggregateWeightedLocalEdges(AggregateLocalEdges):
    def __init__(self, pooling_method="sum", normalize_by_weights=False, **kwargs):
        super(AggregateWeightedLocalEdges, self).__init__(pooling_method=pooling_method, **kwargs)
        self.normalize_by_weights = normalize_by_weights

    def call(self, inputs, **kwargs):
        n, edges, edge_index, weights = inputs
        edges = edges*weights
        # For test only sum scatter, no segment operation etc.
        out = super(AggregateWeightedLocalEdges, self).call([n, edges, edge_index], **kwargs)

        # if self.normalize_by_weights:
        #     norm = tensor_scatter_nd_ops_by_name(
        #         "add", tf.zeros(tf.concat([tf.shape(nodes)[:1], tf.shape(edges)[1:]], axis=0), dtype=edges.dtype),
        #         tf.expand_dims(receive_indices, axis=-1), weights
        #     )
        #     # We could also optionally add tf.eps here.
        #     out = tf.math.divide_no_nan(out, norm)
        return out
