from keras.layers import Layer, GRU
from kgcnn.layers.casting import CastDisjointToBatchedAttributes


class PoolingNodesGRU(Layer):

    def __init__(self, units, **kwargs):
        super(PoolingNodesGRU, self).__init__(**kwargs)
        self.units = units
        self.cast_layer = CastDisjointToBatchedAttributes(return_mask=True)
        self.gru = GRU(units=units)

    def call(self, inputs, **kwargs):
        n, mask = self.cast_layer(inputs)
        out = self.gru(n, mask=mask)
        return out

    def get_config(self):
        config = super(PoolingNodesGRU, self).get_config()
        config.update({"units": self.units})
        return config
