from keras_core.layers import Layer
from keras_core import ops

class Aggregate(Layer):

    def __init__(self, axis=0, **kwargs):
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
    def __init__(self, **kwargs):
        super(AggregateLocalEdges, self).__init__(**kwargs)
        self.to_aggregate = Aggregate()

    def call(self, inputs, **kwargs):
        n, edges, edge_index = inputs
        # For test only sum scatter, no segment operation etc.
        return self.to_aggregate(edges, edge_index[1], ops.shape(n)[0])
