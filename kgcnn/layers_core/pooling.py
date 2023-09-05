import keras_core as ks
from keras_core.layers import Layer
from keras_core import ops


class PoolingNodes(Layer):

    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        super(PoolingNodes, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            return input_shape[1:]
        return tuple(list(input_shape[1][:1]) + list(input_shape[0][1:]))

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return ops.sum(inputs, axis=0)
        x, lengths = inputs
        batch = ops.expand_dims(ops.repeat(ops.arange(ops.shape(lengths)[0], dtype="int32"), lengths), axis=-1)
        return ops.scatter(batch, x, shape=ops.concatenate([ops.shape(lengths)[:1], ops.shape(x)[1:]], axis=0))
