import keras_core as ks
from keras_core.layers import Layer
from keras_core import ops
from kgcnn.layers_core.aggr import Aggregate


class PoolingNodes(Layer):

    def __init__(self, pooling_method="scatter_sum", **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self._to_aggregate = Aggregate(pooling_method=pooling_method)

    def build(self, input_shape):
        self._to_aggregate.build([input_shape[0], (None, ), input_shape[1]])
        self.built = True

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[1][:1]) + list(input_shape[0][1:]))

    def call(self, inputs, **kwargs):
        lengths, x, batch = inputs
        return self._to_aggregate([x, batch, lengths])
