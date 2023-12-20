from keras.layers import Layer, GRU
from kgcnn.layers.casting import CastDisjointToBatchedAttributes


class PoolingNodesGRU(Layer):

    def __init__(self, units, static_output_shape=None,
                 activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros', kernel_regularizer=None,
                 recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                 recurrent_dropout=0.0, reset_after=True, seed=None,
                 **kwargs):
        super(PoolingNodesGRU, self).__init__(**kwargs)
        self.units = units
        self.static_output_shape = static_output_shape
        self.cast_layer = CastDisjointToBatchedAttributes(
            static_output_shape=static_output_shape, return_mask=True)
        self.gru = GRU(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            seed=seed
        )

    def call(self, inputs, **kwargs):
        n, mask = self.cast_layer(inputs)
        out = self.gru(n, mask=mask)
        return out

    def get_config(self):
        config = super(PoolingNodesGRU, self).get_config()
        config.update({"units": self.units, "static_output_shape": self.static_output_shape})
        conf_gru = self.gru.get_config()
        param_list = ["units", "activation", "recurrent_activation",
                      "use_bias", "kernel_initializer",
                      "recurrent_initializer",
                      "bias_initializer", "kernel_regularizer",
                      "recurrent_regularizer", "bias_regularizer", "kernel_constraint",
                      "recurrent_constraint", "bias_constraint", "dropout",
                      "recurrent_dropout", "reset_after"]
        for x in param_list:
            if x in conf_gru.keys():
                config.update({x: conf_gru[x]})
        return config
