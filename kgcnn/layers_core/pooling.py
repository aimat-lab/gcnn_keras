import keras_core as ks
from keras_core.layers import Layer
from keras_core import ops


class PoolingNodes(Layer):

    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return ops.sum(inputs, axis=0)
        x, batch = inputs
        return ops.scatter(batch, x, shape=ops.concatenate([[ops.max(batch)], ops.shape(x)[1:]], axis=0))
