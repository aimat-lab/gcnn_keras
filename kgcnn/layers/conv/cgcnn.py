from kgcnn.layers.conv.message import MessagePassingBase
from kgcnn.layers.norm import GraphBatchNormalization

import tensorflow.keras as ks
import tensorflow as tf

class CGCNNLayer(MessagePassingBase):

    def __init__(self, units=64, activation_s='relu', activation_out='relu', batch_normalization=False, **kwargs):
        """Initialize MessagePassingBase layer."""
        super(CGCNNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_f = ks.activations.sigmoid
        self.activation_s = ks.layers.Activation(activation_s)
        self.activation_out = ks.layers.Activation(activation_out)
        self.batch_norm_f = GraphBatchNormalization()
        self.batch_norm_s = GraphBatchNormalization()
        self.batch_norm_out = GraphBatchNormalization()
        self.f = ks.layers.Dense(self.units)
        self.s = ks.layers.Dense(self.units)
        self.batch_normalization = batch_normalization

    def message_function(self, inputs, **kwargs):

        nodes_in = inputs[0] # shape: (batch_size, M, F)
        nodes_out = inputs[1] # shape: (batch_size, M, F)
        edge_features = inputs[2] # shape: (batch_size, M, E)

        x = tf.concat([nodes_in, nodes_out, edge_features], axis=2)
        x_s, x_f = self.s(x), self.f(x)
        if self.batch_normalization:
            x_s, x_f = self.batch_norm_s(x_s), self.batch_norm_f(x_f)
        x_s, x_f = self.activation_s(x_s), self.activation_f(x_f)
        x_out = x_s * x_f # shape: (batch_size, M, self.units)
        if self.batch_normalization:
            x_out = self.batch_norm_out(x_out)
        return x_out

    def update_nodes(self, inputs, **kwargs):
        nodes = inputs[0]
        nodes_update = inputs[1]
        return nodes + nodes_update
